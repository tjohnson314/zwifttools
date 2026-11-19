import * as Common from '/pages/src/common.mjs';
import { DETAIL_DEFAULTS } from './details.mjs';

Common.enableSentry();

export function settingsMain() {
    Common.initInteractionListeners();
    Common.settingsStore.setDefault({...DETAIL_DEFAULTS});
    Common.initSettingsForm('form#options')();
}
