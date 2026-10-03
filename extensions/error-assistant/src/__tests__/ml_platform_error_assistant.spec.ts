import { ServerConnection } from '@jupyterlab/services';

import { applyFixToCellModel } from '../fix-utils';
import { requestAPI } from '../request';

describe('@ml-platform/error-assistant', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('uses the Jupyter base URL and returns the server JSON response', async () => {
    const settings = { baseUrl: '/user/test/' } as ServerConnection.ISettings;
    const response = {
      ok: true,
      text: async () => JSON.stringify({ status: 'ok' })
    } as Response;
    const makeRequest = jest
      .spyOn(ServerConnection, 'makeRequest')
      .mockResolvedValue(response);

    await expect(requestAPI('api/health', settings)).resolves.toEqual({
      status: 'ok'
    });
    expect(makeRequest).toHaveBeenCalledWith(
      '/user/test/api/health',
      {},
      settings
    );
  });

  it('writes the suggested code through the cell shared model', () => {
    const setSource = jest.fn();
    const model = { sharedModel: { setSource } };

    expect(applyFixToCellModel(model, 'import matplotlib.pyplot as plt')).toBe(
      true
    );
    expect(setSource).toHaveBeenCalledWith('import matplotlib.pyplot as plt');
  });

  it('rejects a missing cell model or empty code suggestion', () => {
    expect(applyFixToCellModel(null, 'print(1)')).toBe(false);
    expect(
      applyFixToCellModel({ sharedModel: { setSource: jest.fn() } }, ' ')
    ).toBe(false);
  });
});
