export function applyFixToCellModel(
  model: { sharedModel?: { setSource?: (source: string) => void } } | null,
  fixText: string
): boolean {
  const sharedModel = model?.sharedModel;
  const source = fixText.trim();
  if (!sharedModel || typeof sharedModel.setSource !== 'function' || !source) {
    return false;
  }

  sharedModel.setSource(source);
  return true;
}
