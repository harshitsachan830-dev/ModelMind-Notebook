import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import { isCodeCellModel } from '@jupyterlab/cells';
import { IMainMenu } from '@jupyterlab/mainmenu';
import type { IError, IOutput } from '@jupyterlab/nbformat';
import { INotebookTracker, NotebookActions } from '@jupyterlab/notebook';
import { Widget } from '@lumino/widgets';

import modelMindLogo from '../style/modelmind-logo.png';

import { applyFixToCellModel } from './fix-utils';
import { requestAPI } from './request';

const COMMAND_ID = '@ml-platform/error-assistant:open';
const MAX_CODE_LENGTH = 20_000;
const MAX_TRACEBACK_LENGTH = 12_000;

function setModelMindFavicon(): void {
  document.querySelectorAll('link[rel~="icon"]').forEach(link => {
    link.remove();
  });

  const favicon = document.createElement('link');
  favicon.rel = 'icon';
  favicon.type = 'image/png';
  favicon.setAttribute('sizes', '256x256');
  favicon.href = modelMindLogo;
  document.head.appendChild(favicon);
}

type CapturedError = {
  cell_id: string;
  code: string;
  error_type: string;
  error_message: string;
  traceback: string;
};

type NotebookCell = NonNullable<INotebookTracker['activeCell']>;

type FixCandidate = {
  summary: string;
  candidate_code: string;
  provider?: 'ollama' | 'gemini';
};

type FixActions = {
  suggestFix: () => Promise<FixCandidate>;
  applyFix: (fix: FixCandidate) => Promise<void>;
};

function getTargetCell(notebooks: INotebookTracker): NotebookCell | null {
  return (
    notebooks.activeCell ?? notebooks.currentWidget?.content.activeCell ?? null
  );
}

function captureActiveError(cell: NotebookCell | null): CapturedError | null {
  if (!cell || !isCodeCellModel(cell.model)) {
    return null;
  }

  let errorOutput: IError | undefined;
  for (const output of cell.model.outputs.toJSON().reverse()) {
    if (isErrorOutput(output)) {
      errorOutput = output;
      break;
    }
  }
  if (!errorOutput) {
    return null;
  }

  const source = cell.model.toJSON().source;
  const code = Array.isArray(source) ? source.join('') : source;
  return {
    cell_id: cell.model.id.slice(0, 256),
    code: code.slice(0, MAX_CODE_LENGTH),
    error_type: errorOutput.ename.slice(0, 200),
    error_message: errorOutput.evalue.slice(0, 2_000),
    traceback: errorOutput.traceback.join('\n').slice(0, MAX_TRACEBACK_LENGTH)
  };
}

function getAdjacentNotebookCode(
  notebook: NonNullable<INotebookTracker['currentWidget']>,
  targetCell: NotebookCell
): string {
  const cells = notebook.content.widgets;
  const targetIndex = cells.indexOf(targetCell);
  if (targetIndex < 0) {
    return '';
  }

  return cells
    .slice(Math.max(0, targetIndex - 1), targetIndex + 2)
    .filter(cell => cell !== targetCell && isCodeCellModel(cell.model))
    .map(cell => {
      const source = cell.model.toJSON().source;
      return Array.isArray(source) ? source.join('') : source;
    })
    .join('\n')
    .slice(0, 12_000);
}

function isErrorOutput(output: IOutput): output is IError {
  return (
    output.output_type === 'error' &&
    'ename' in output &&
    typeof output.ename === 'string' &&
    'evalue' in output &&
    typeof output.evalue === 'string' &&
    'traceback' in output &&
    Array.isArray(output.traceback) &&
    output.traceback.every((line: unknown) => typeof line === 'string')
  );
}

function createStatusPill(label: string, tone: 'default' | 'success' | 'warn') {
  const pill = document.createElement('div');
  pill.className = `ml-assistant-status-pill ml-assistant-status-${tone}`;
  pill.textContent = label;
  return pill;
}

function renderAssistantResult(
  container: HTMLElement,
  payload: {
    errorType: string;
    errorMessage: string;
    traceback: string;
    summary: string;
    details: string[];
    fix: string;
  },
  fixActions?: FixActions,
  suggestImmediately = false
) {
  container.replaceChildren();

  const header = document.createElement('div');
  header.className = 'ml-assistant-result-header';

  const banner = document.createElement('div');
  banner.className = 'ml-assistant-error-banner';
  const errorIcon = document.createElement('span');
  errorIcon.className = 'ml-assistant-error-icon';
  errorIcon.setAttribute('aria-hidden', 'true');
  errorIcon.textContent = '!';
  const errorText = document.createElement('div');
  errorText.className = 'ml-assistant-error-copy';
  const errorTitle = document.createElement('div');
  errorTitle.className = 'ml-assistant-error-title';
  errorTitle.textContent = payload.errorType;
  const errorMessage = document.createElement('div');
  errorMessage.className = 'ml-assistant-error-message';
  errorMessage.textContent = payload.errorMessage;
  errorText.append(errorTitle, errorMessage);
  banner.append(errorIcon, errorText);

  const traceback = document.createElement('div');
  traceback.className = 'ml-assistant-traceback';
  const tracebackDetails = document.createElement('details');
  const tracebackSummary = document.createElement('summary');
  tracebackSummary.textContent = 'View traceback';
  const tracebackContent = document.createElement('pre');
  tracebackContent.textContent = payload.traceback;
  tracebackDetails.append(tracebackSummary, tracebackContent);
  traceback.append(tracebackDetails);

  const explanation = document.createElement('div');
  explanation.className = 'ml-assistant-card ml-assistant-card-blue';
  const explanationTitle = document.createElement('div');
  explanationTitle.className = 'ml-assistant-card-title';
  explanationTitle.textContent = 'Explanation';
  const explanationSummary = document.createElement('p');
  explanationSummary.textContent = payload.summary;
  const explanationDetails = document.createElement('ul');
  payload.details.forEach(detail => {
    const item = document.createElement('li');
    item.textContent = detail;
    explanationDetails.appendChild(item);
  });
  explanation.append(explanationTitle, explanationSummary, explanationDetails);

  const fixCard = document.createElement('div');
  fixCard.className = 'ml-assistant-card ml-assistant-card-green';
  const fixTitle = document.createElement('div');
  fixTitle.className = 'ml-assistant-card-title';
  fixTitle.textContent = 'Possible solution';
  const fixSummary = document.createElement('p');
  fixSummary.textContent = payload.fix;
  const fixCode = document.createElement('div');
  fixCode.className = 'ml-assistant-code-block';
  fixCode.textContent = 'Generate a suggestion to see corrected code here.';
  let currentFix: FixCandidate | null = null;
  fixCard.append(fixTitle, fixSummary, fixCode);

  const actions = document.createElement('div');
  actions.className = 'ml-assistant-actions';
  const suggestButton = document.createElement('button');
  suggestButton.type = 'button';
  suggestButton.textContent = 'Suggest Fix';
  suggestButton.disabled = !fixActions;
  const applyButton = document.createElement('button');
  applyButton.type = 'button';
  applyButton.textContent = 'Apply Fix';
  applyButton.disabled = !fixActions;
  const copyButton = document.createElement('button');
  copyButton.type = 'button';
  copyButton.textContent = 'Copy Code';
  copyButton.className = 'ml-assistant-secondary-button';

  const generateFix = async (): Promise<FixCandidate> => {
    if (!fixActions) {
      throw new Error('No fix action is available');
    }
    if (currentFix) {
      return currentFix;
    }
    suggestButton.disabled = true;
    applyButton.disabled = true;
    suggestButton.textContent = 'Generating...';
    try {
      currentFix = await fixActions.suggestFix();
      fixSummary.textContent =
        currentFix.provider === 'gemini'
          ? `Gemini suggestion: ${currentFix.summary}`
          : currentFix.summary;
      fixCode.textContent = currentFix.candidate_code;
      copyButton.disabled = false;
      return currentFix;
    } finally {
      suggestButton.textContent = 'Suggest Fix';
      suggestButton.disabled = false;
      applyButton.disabled = false;
    }
  };

  suggestButton.addEventListener('click', async () => {
    suggestButton.disabled = true;
    suggestButton.textContent = 'Generating...';
    try {
      await generateFix();
    } catch (error) {
      fixSummary.textContent =
        error instanceof Error ? error.message : 'Could not generate a fix';
    } finally {
      suggestButton.disabled = false;
      suggestButton.textContent = 'Suggest Fix';
    }
  });

  applyButton.addEventListener('click', async () => {
    if (!fixActions) {
      return;
    }
    applyButton.disabled = true;
    suggestButton.disabled = true;
    applyButton.textContent = currentFix ? 'Applying...' : 'Generating...';
    try {
      const fix = await generateFix();
      await fixActions.applyFix(fix);
      applyButton.textContent = 'Fixed and ran';
    } catch (error) {
      applyButton.textContent =
        error instanceof Error ? error.message : 'Could not apply fix';
    } finally {
      suggestButton.disabled = false;
    }
    window.setTimeout(() => {
      applyButton.textContent = 'Apply Fix';
      applyButton.disabled = false;
    }, 2500);
  });

  copyButton.addEventListener('click', async () => {
    if (!currentFix) {
      return;
    }
    try {
      await navigator.clipboard.writeText(currentFix.candidate_code);
      copyButton.textContent = 'Copied';
    } catch {
      copyButton.textContent = 'Copy failed';
    }

    window.setTimeout(() => {
      copyButton.textContent = 'Copy Code';
    }, 1200);
  });

  copyButton.disabled = true;
  suggestButton.classList.add('ml-assistant-action-wide');
  actions.append(suggestButton, applyButton, copyButton);

  header.append(banner, traceback);
  container.append(header, explanation, fixCard, actions);
  if (suggestImmediately && fixActions) {
    suggestButton.click();
  }
}

const plugin: JupyterFrontEndPlugin<void> = {
  id: '@ml-platform/error-assistant:plugin',
  description: 'JupyterLab error assistant frontend and server extension',
  autoStart: true,
  requires: [IMainMenu, INotebookTracker],
  activate: (
    app: JupyterFrontEnd,
    mainMenu: IMainMenu,
    notebooks: INotebookTracker
  ) => {
    setModelMindFavicon();

    const panel = new Widget();
    panel.id = 'ml-platform-error-assistant-panel';
    panel.title.label = 'Error Assistant';
    panel.title.caption = 'Notebook error assistant';
    panel.addClass('ml-platform-error-assistant-panel');

    const shell = document.createElement('div');
    shell.className = 'ml-assistant-shell';

    const header = document.createElement('div');
    header.className = 'ml-assistant-header';
    const heading = document.createElement('h2');
    heading.textContent = 'AI Assistant';
    const controls = document.createElement('div');
    controls.className = 'ml-assistant-header-controls';
    const installButton = document.createElement('button');
    installButton.type = 'button';
    installButton.className = 'ml-assistant-primary-button';
    installButton.textContent = 'Install Ollama';
    installButton.disabled = true;
    const cloudOption = document.createElement('label');
    cloudOption.className = 'ml-assistant-cloud-option';
    cloudOption.title =
      'When enabled, the failed cell, traceback, and nearby code may be sent to Google Gemini, either directly or as a fallback.';
    const geminiFallbackToggle = document.createElement('input');
    geminiFallbackToggle.type = 'checkbox';
    geminiFallbackToggle.disabled = true;
    const cloudOptionText = document.createElement('span');
    cloudOptionText.textContent = 'Allow Gemini use';
    cloudOption.append(geminiFallbackToggle, cloudOptionText);

    const statusRow = document.createElement('div');
    statusRow.className = 'ml-assistant-status-row';
    const localStatus = createStatusPill('Ollama (Local AI)', 'success');
    const cloudStatus = createStatusPill('Cloud disabled', 'default');
    const providerRow = document.createElement('label');
    providerRow.className = 'ml-assistant-provider-row';
    providerRow.textContent = 'Fix provider';
    const providerSelect = document.createElement('select');
    providerSelect.className = 'ml-assistant-provider-select';
    const ollamaOption = document.createElement('option');
    ollamaOption.value = 'ollama';
    ollamaOption.textContent = 'Ollama (local)';
    const geminiOption = document.createElement('option');
    geminiOption.value = 'gemini';
    geminiOption.textContent = 'Gemini (cloud)';
    providerSelect.append(ollamaOption, geminiOption);
    providerRow.appendChild(providerSelect);

    const tabRow = document.createElement('div');
    tabRow.className = 'ml-assistant-tab-row';
    const tabs = ['Explain Error', 'Suggest Fix', 'Code Help', 'Ask AI'];
    const tabButtons = new Map<string, HTMLButtonElement>();
    tabs.forEach((tabText, index) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = `ml-assistant-tab ${index === 0 ? 'active' : ''}`;
      btn.textContent = tabText;
      tabRow.appendChild(btn);
      tabButtons.set(tabText, btn);
    });

    const ctaWrap = document.createElement('div');
    ctaWrap.className = 'ml-assistant-cta-row';
    const captureButton = document.createElement('button');
    captureButton.type = 'button';
    captureButton.className = 'ml-assistant-cta-button';
    captureButton.textContent = 'Analyze Error';
    ctaWrap.appendChild(captureButton);

    const content = document.createElement('div');
    content.className = 'ml-assistant-content';

    const makePlaceholder = () => {
      const empty = document.createElement('div');
      empty.className = 'ml-assistant-empty';
      empty.textContent = 'Select a failed code cell to capture its error.';
      return empty;
    };

    let analysisMode: 'explain' | 'suggest' = 'explain';

    content.appendChild(makePlaceholder());

    header.append(heading, controls);
    controls.append(installButton, cloudOption);
    statusRow.append(localStatus, cloudStatus);
    shell.append(header, statusRow, providerRow, tabRow, ctaWrap, content);
    panel.node.append(shell);

    const setStatus = async () => {
      try {
        const health = await requestAPI<{
          status: string;
          cloud_fallback: string;
          gemini_fallback_available: boolean;
        }>('api/health', app.serviceManager.serverSettings);
        localStatus.textContent =
          health.status === 'ok' ? 'Ollama (Local AI)' : 'Platform unavailable';
        localStatus.className =
          health.status === 'ok'
            ? 'ml-assistant-status-pill ml-assistant-status-success'
            : 'ml-assistant-status-pill ml-assistant-status-warn';
        geminiFallbackToggle.disabled = !health.gemini_fallback_available;
        geminiOption.disabled = !health.gemini_fallback_available;
        if (geminiOption.disabled && providerSelect.value === 'gemini') {
          providerSelect.value = 'ollama';
        }
        cloudStatus.textContent = health.gemini_fallback_available
          ? 'Gemini available (opt-in)'
          : health.cloud_fallback === 'enabled'
            ? 'Gemini API key missing'
            : 'Cloud disabled by server';
        cloudStatus.className = health.gemini_fallback_available
          ? 'ml-assistant-status-pill ml-assistant-status-success'
          : 'ml-assistant-status-pill ml-assistant-status-default';
      } catch {
        localStatus.textContent = 'Platform unavailable';
        localStatus.className =
          'ml-assistant-status-pill ml-assistant-status-warn';
        geminiFallbackToggle.disabled = true;
        cloudStatus.textContent = 'Cloud status unavailable';
        cloudStatus.className =
          'ml-assistant-status-pill ml-assistant-status-warn';
      }

      try {
        const ollama = await requestAPI<{
          status: string;
          version: string | null;
          model_status: 'available' | 'missing' | 'unavailable' | 'not_checked';
          model_id: string;
        }>('api/local-ai/status', app.serviceManager.serverSettings);
        if (
          ollama.status === 'connected' &&
          ollama.model_status === 'available'
        ) {
          localStatus.textContent = 'Ollama (Local AI)';
          localStatus.className =
            'ml-assistant-status-pill ml-assistant-status-success';
          installButton.disabled = true;
        } else if (ollama.status === 'connected') {
          localStatus.textContent = 'Model missing';
          localStatus.className =
            'ml-assistant-status-pill ml-assistant-status-warn';
          installButton.disabled = false;
        }
      } catch {
        localStatus.textContent = 'Local AI unavailable';
        localStatus.className =
          'ml-assistant-status-pill ml-assistant-status-warn';
        installButton.disabled = true;
      }
    };

    captureButton.addEventListener('click', async () => {
      const targetCell = getTargetCell(notebooks);
      const notebookPanel = notebooks.currentWidget;
      const error = captureActiveError(targetCell);
      if (!error) {
        renderAssistantResult(content, {
          errorType: 'No error detected',
          errorMessage: 'The active code cell has no error output.',
          traceback: 'No traceback available.',
          summary: 'Select a cell that produced an actual runtime error.',
          details: ['The selected cell is not currently failing.'],
          fix: 'print("Select a failing cell and retry.")'
        });
        return;
      }

      content.innerHTML = '';
      captureButton.disabled = true;
      const loading = document.createElement('div');
      loading.className = 'ml-assistant-empty';
      loading.textContent = 'Capturing error...';
      content.appendChild(loading);

      try {
        const result = await requestAPI<{
          status: string;
          analysis: {
            summary: string;
            details: string[];
            is_sufficient: boolean;
          };
        }>('api/error/analyze', app.serviceManager.serverSettings, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(error)
        });

        if (result.status !== 'analyzed') {
          throw new Error('analysis unavailable');
        }

        let explanationSummary = result.analysis.summary;
        let explanationDetails = result.analysis.details;
        const suggestFix = async (): Promise<FixCandidate> => {
          if (!targetCell || !notebookPanel) {
            throw new Error('Select a notebook cell first');
          }

          const suggestion = await requestAPI<{
            status: string;
            summary?: string;
            candidate_code?: string;
            model_id?: string;
            provider?: 'ollama' | 'gemini';
            message?: string;
          }>('api/ai/fix', app.serviceManager.serverSettings, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              error,
              notebook_context: getAdjacentNotebookCode(
                notebookPanel,
                targetCell
              ),
              provider: providerSelect.value,
              allow_gemini_fallback: geminiFallbackToggle.checked
            })
          });
          if (suggestion.status !== 'suggested' || !suggestion.candidate_code) {
            throw new Error(
              suggestion.message ||
                (suggestion.status === 'gemini_disabled'
                  ? 'Gemini fallback is disabled by server policy'
                  : suggestion.status === 'gemini_opt_in_required'
                    ? 'Check Allow Gemini use before selecting Gemini'
                    : suggestion.status === 'gemini_unconfigured'
                      ? 'Configure the Gemini API key on the server first'
                      : suggestion.status === 'gemini_unavailable'
                        ? 'Gemini is temporarily unavailable; try again later'
                        : suggestion.status === 'model_missing'
                          ? `Install ${suggestion.model_id || 'the local Ollama model'} first`
                          : suggestion.status === 'incomplete_model_response' ||
                              suggestion.status === 'incomplete_gemini_response'
                            ? 'The AI returned only part of the cell. Nothing was applied; try again or enable Gemini fallback.'
                            : 'Neither local Ollama nor the enabled Gemini fallback produced a valid fix')
            );
          }

          return {
            summary: suggestion.summary || 'A fix candidate is ready.',
            candidate_code: suggestion.candidate_code,
            provider: suggestion.provider || 'ollama'
          };
        };

        const applyFix = async (suggestion: FixCandidate): Promise<void> => {
          if (!targetCell || !notebookPanel) {
            throw new Error('Select a notebook cell first');
          }
          if (
            !applyFixToCellModel(targetCell.model, suggestion.candidate_code)
          ) {
            throw new Error('Could not update the failed cell');
          }

          const executed = await NotebookActions.runCells(
            notebookPanel.content,
            [targetCell],
            notebookPanel.sessionContext
          );
          if (!executed) {
            throw new Error('Fix applied; cell could not be run');
          }
          if (
            !isCodeCellModel(targetCell.model) ||
            targetCell.model.outputs.toJSON().some(isErrorOutput)
          ) {
            throw new Error('Fix applied, but the cell still reports an error');
          }
        };
        const fixActions = { suggestFix, applyFix };

        if (result.analysis.is_sufficient) {
          renderAssistantResult(
            content,
            {
              errorType: error.error_type,
              errorMessage: error.error_message,
              traceback: error.traceback,
              summary: explanationSummary,
              details: explanationDetails,
              fix: 'A corrected cell will be generated locally when you apply the fix.'
            },
            fixActions,
            analysisMode === 'suggest'
          );
          return;
        }

        const explanation = await requestAPI<{
          status: string;
          model_id?: string;
          explanation?: {
            summary: string;
            details: string[];
            confidence: string;
          };
        }>('api/ai/explain', app.serviceManager.serverSettings, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(error)
        });

        if (explanation.status === 'explained' && explanation.explanation) {
          explanationSummary = explanation.explanation.summary;
          explanationDetails = explanation.explanation.details;
        }

        renderAssistantResult(
          content,
          {
            errorType: error.error_type,
            errorMessage: error.error_message,
            traceback: error.traceback,
            summary: explanationSummary,
            details: explanationDetails,
            fix: 'A corrected cell will be generated locally when you apply the fix.'
          },
          fixActions,
          analysisMode === 'suggest'
        );
      } catch {
        renderAssistantResult(content, {
          errorType: 'Error analysis unavailable',
          errorMessage: 'The platform could not diagnose this error.',
          traceback: 'The backend returned no valid analysis.',
          summary: 'The local platform could not validate the notebook error.',
          details: ['Check the notebook output and retry the analysis.'],
          fix: 'print("Review the failing cell and confirm the variable or column exists.")'
        });
      } finally {
        captureButton.disabled = false;
      }
    });

    const showCodeHelp = (): void => {
      const targetCell = getTargetCell(notebooks);
      if (!targetCell || !isCodeCellModel(targetCell.model)) {
        const message = document.createElement('div');
        message.className = 'ml-assistant-empty';
        message.textContent =
          'Select a code cell to ask local Ollama about it.';
        content.replaceChildren(message);
        return;
      }

      const source = targetCell.model.toJSON().source;
      const code = (Array.isArray(source) ? source.join('') : source).slice(
        0,
        MAX_CODE_LENGTH
      );
      const notebookPanel = notebooks.currentWidget;
      const questionInput = document.createElement('textarea');
      questionInput.className = 'ml-assistant-question-input';
      questionInput.maxLength = 1_000;
      questionInput.value = 'Explain this cell and point out potential issues.';
      const questionLabel = document.createElement('label');
      questionLabel.className = 'ml-assistant-question-label';
      questionLabel.textContent = 'Ask a question about this cell';
      questionLabel.appendChild(questionInput);

      const askButton = document.createElement('button');
      askButton.type = 'button';
      askButton.className = 'ml-assistant-cta-button';
      askButton.textContent = 'Ask Local Ollama';
      const answer = document.createElement('div');
      answer.className = 'ml-assistant-code-help-answer';
      content.replaceChildren(questionLabel, askButton, answer);

      askButton.addEventListener('click', async () => {
        const question = questionInput.value.trim();
        if (!question) {
          answer.textContent = 'Enter a question about this code first.';
          return;
        }

        askButton.disabled = true;
        answer.textContent = 'Asking local Ollama...';
        try {
          const result = await requestAPI<{
            status: string;
            model_id?: string;
            explanation?: {
              summary: string;
              details: string[];
            };
          }>('api/ai/code-help', app.serviceManager.serverSettings, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              code,
              question,
              notebook_context:
                notebookPanel && targetCell
                  ? getAdjacentNotebookCode(notebookPanel, targetCell)
                  : ''
            })
          });

          answer.replaceChildren();
          if (result.status !== 'explained' || !result.explanation) {
            answer.textContent =
              result.status === 'model_missing'
                ? `Install ${result.model_id || 'the local Ollama model'} first.`
                : 'Local Ollama could not answer this code question.';
            return;
          }

          const summary = document.createElement('p');
          summary.textContent = result.explanation.summary;
          const details = document.createElement('ul');
          result.explanation.details.forEach(detail => {
            const item = document.createElement('li');
            item.textContent = detail;
            details.appendChild(item);
          });
          answer.append(summary, details);
        } catch {
          answer.textContent = 'Local code help is unavailable right now.';
        } finally {
          askButton.disabled = false;
        }
      });
    };

    tabButtons.forEach((tabButton, tabText) => {
      tabButton.addEventListener('click', () => {
        tabButtons.forEach(button => button.classList.remove('active'));
        tabButton.classList.add('active');
        if (tabText === 'Code Help' || tabText === 'Ask AI') {
          analysisMode = 'explain';
          showCodeHelp();
          return;
        }
        analysisMode = tabText === 'Suggest Fix' ? 'suggest' : 'explain';
        void captureButton.click();
      });
    });

    installButton.addEventListener('click', async () => {
      installButton.disabled = true;
      try {
        const result = await requestAPI<{
          status: string;
          model_id?: string;
          message?: string;
        }>('api/local-ai/install', app.serviceManager.serverSettings, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ allow_download: true })
        });
        if (result.status === 'installed') {
          await setStatus();
          renderAssistantResult(content, {
            errorType: 'Model installed',
            errorMessage: result.message || 'Local model ready',
            traceback: 'Installation completed successfully.',
            summary:
              'The recommended local model is available for notebook error explanations.',
            details: ['The local Ollama path is now active.'],
            fix: 'print("Model ready. Re-run the failing cell to explain the error.")'
          });
          return;
        }
      } catch {
        // no-op; status remains visible to user
      } finally {
        await setStatus();
      }
    });

    app.shell.add(panel, 'right', { rank: 900 });
    app.commands.addCommand(COMMAND_ID, {
      label: 'Open Error Assistant',
      execute: () => {
        app.shell.activateById(panel.id);
        void setStatus();
      }
    });
    const helpMenu = mainMenu.helpMenu as unknown as Widget;
    mainMenu.viewMenu.addGroup([{ command: COMMAND_ID }], 900);
    helpMenu.hide();
    void setStatus();
  }
};

export default plugin;
