/** Preference key for the WebGPU→CPU fallback dialog. Kept outside Progress. */
export const BACKEND_FALLBACK_DIALOG_KEY = 'differentiable.backendFallbackDialogDismissed';

/** True when the player previously chose "Don't show again". */
export function isBackendFallbackDialogDismissed(
  storage: Pick<Storage, 'getItem'> | null | undefined = safeLocalStorage(),
): boolean {
  if (!storage) return false;
  try {
    return storage.getItem(BACKEND_FALLBACK_DIALOG_KEY) === '1';
  } catch {
    return false;
  }
}

/** Persist "Don't show again" so the CPU-fallback dialog stays closed. */
export function dismissBackendFallbackDialog(
  storage: Pick<Storage, 'setItem'> | null | undefined = safeLocalStorage(),
): void {
  if (!storage) return;
  try {
    storage.setItem(BACKEND_FALLBACK_DIALOG_KEY, '1');
  } catch {
    // Preferences are a convenience; the game runs fine without them.
  }
}

function safeLocalStorage(): Storage | undefined {
  try {
    return globalThis.localStorage;
  } catch {
    return undefined;
  }
}
