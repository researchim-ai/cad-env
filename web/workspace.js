/* An illustrative protocol explorer. No remote inference or CAD execution. */
(() => {
  'use strict';
  const next = document.querySelector('#loop-next');
  if (!next) return;
  const phases = [
    {
      direction: 'ENV → AGENT',
      code: '{\n  "goal": "Set the marked hole to Ø 8 mm",\n  "observation": ["goal image", "public CAD state"]\n}',
      en: 'The model sees the task and current CAD state. The reference model is kept by the evaluator.',
      ru: 'Модель видит задачу и текущее состояние CAD. Эталонная модель хранится у проверяющего.'
    },
    {
      direction: 'AGENT → TOOL',
      code: '{\n  "op": "set_parameter",\n  "target": "Holes",\n  "parameter": "diameter_0", "value": 8\n}',
      en: 'The model returns one JSON action. Here it changes a named diameter to 8 mm.',
      ru: 'Модель возвращает одно JSON-действие. В этом примере оно меняет именованный диаметр на 8 мм.'
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
      en: 'After finish, the evaluator checks the saved part. This is an example of the interface, not a recorded model run.',
      ru: 'После finish проверяющий оценивает сохранённую деталь. Здесь показан пример взаимодействия, а не записанная попытка модели.'
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
