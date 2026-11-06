/* An illustrative protocol explorer. No remote inference or CAD execution. */
(() => {
  'use strict';
  const next = document.querySelector('#loop-next');
  if (!next) return;
  const phases = [
    {
      direction: 'ENV → AGENT',
      code: '{\n  "goal": "Set the marked hole to Ø 8 mm",\n  "observation": ["goal image", "public CAD state"]\n}',
      en: 'The model receives a goal and public observations. Hidden reference geometry belongs to the evaluator, never the policy input.',
      ru: 'Модель получает задачу и публичные наблюдения. Скрытая эталонная геометрия доступна оценщику, но не передаётся агенту.'
    },
    {
      direction: 'AGENT → TOOL',
      code: '{\n  "op": "set_parameter",\n  "target": "Holes",\n  "parameter": "diameter_0", "value": 8\n}',
      en: 'One typed action changes a model parameter. Tool names and JSON stay the same across languages; the surrounding task can be multimodal.',
      ru: 'Одно типизированное действие меняет параметр модели. Названия инструментов и JSON одинаковы в обоих языках; сама задача может быть мультимодальной.'
    },
    {
      direction: 'TOOL → CAD KERNEL',
      code: '{\n  "operation": "execute & recompute",\n  "returns": ["updated state", "execution errors"],\n  "optional": "current candidate renders"\n}',
      en: 'Native execution updates the actual CAD document. Errors become feedback. Current renders are optional and depend on the observation profile.',
      ru: 'Нативное исполнение обновляет настоящий CAD-документ. Ошибки возвращаются как обратная связь. Текущие рендеры зависят от профиля наблюдений.'
    },
    {
      direction: 'ENV → LEARNING SYSTEM',
      code: '{\n  "checks": ["geometry", "protected parameters",\n             "feature history", "task success"],\n  "reward": "computed by the native evaluator"\n}',
      en: 'A finish action triggers native task validation. This diagram illustrates the protocol; it is not an episode result or a claim of RL improvement.',
      ru: 'Действие finish запускает нативную проверку задачи. Это схема протокола, а не результат эпизода или заявление об улучшении от RL.'
    }
  ];
  let phase = 0;
  function render() {
    const current = phases[phase];
    const ru = window.CAD_I18N?.locale === 'ru';
    document.querySelector('#loop-direction').textContent = current.direction;
    document.querySelector('#loop-index').textContent = `0${phase + 1} / 04`;
    document.querySelector('#loop-code').textContent = current.code;
    document.querySelector('#loop-description').textContent = current[ru ? 'ru' : 'en'];
    document.querySelectorAll('[data-phase]').forEach(button => button.setAttribute('aria-pressed', String(Number(button.dataset.phase) === phase)));
  }
  document.querySelectorAll('[data-phase]').forEach(button => button.addEventListener('click', () => { phase = Number(button.dataset.phase); render(); }));
  next.addEventListener('click', () => { phase = (phase + 1) % phases.length; render(); });
  document.addEventListener('cad-languagechange', render);
  render();
})();
