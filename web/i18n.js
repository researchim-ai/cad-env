'use strict';
(() => {
  let locale = document.documentElement.lang === 'ru' ? 'ru' : 'en';
  const requested = new URLSearchParams(location.search).get('lang');
  if (requested === 'en' || requested === 'ru') { locale = requested; try { localStorage.setItem('cad-env-language', locale); } catch {} }
  let strings = {};
  const ui = {
    loading: ['Loading verified example…', 'Загрузка проверенного примера…'],
    shaderError: ['3D shader initialization failed', 'Не удалось инициализировать 3D-шейдер'],
    invalidMesh: ['Invalid reference mesh', 'Некорректная эталонная сетка'],
    mainTitle: ['cad-env / A training ground for CAD agents.', 'cad-env / Полигон для CAD-агентов.'],
    guideTitle: ['Field guide / cad-env', 'Руководство / cad-env'],
    mainDescription: ['Native CAD environments, verified multimodal data and training infrastructure for tool-using language models.', 'Нативные CAD-среды, проверенные мультимодальные данные и обучение языковых моделей работе с инструментами.'],
    guideDescription: ['Architecture, native setup and reproducibility boundaries of cad-env.', 'Архитектура cad-env, настройка нативной среды и границы воспроизводимости.'],
    copied: ['Copied to clipboard', 'Скопировано в буфер обмена'],
    clipboard: ['Clipboard unavailable. Select the code to copy.', 'Буфер обмена недоступен. Выделите код для копирования.'],
    bolt_flange: ['Bolt flange', 'Фланец'],
    counterbored_plate: ['Counterbored plate', 'Плита с цековками'],
    stepped_bore: ['Stepped bore', 'Ступенчатое отверстие'],
    nativeReference: ['FreeCAD · validated TRAIN reference', 'FreeCAD · проверенный эталон из TRAIN'],
    actionCount: ['{count} ACTIONS · NATIVE-VERIFIED TEACHER', '{count} ДЕЙСТВИЙ · НАТИВНО ПРОВЕРЕННЫЙ УЧИТЕЛЬ'],
    sequential: ['TASKS · SEQUENTIAL GREEDY EVALUATION', 'ЗАДАЧ · ПОСЛЕДОВАТЕЛЬНЫЙ GREEDY-РЕЖИМ'],
    simple: ['Simple tasks', 'Базовые задачи'], complex: ['Complex builds', 'Сложные построения'],
    grounding: ['Visual selection', 'Визуальный выбор'], entities: ['Entity-labelled selection', 'Выбор с подписями объектов'],
    batch: ['Batch ≤4 · 8 episode workers', 'Пакет ≤4 · 8 параллельных сред'],
    complete: ['Validation complete; retention uses the batch execution profile.', 'Валидация завершена; сохранение навыков проверено в пакетном режиме.'],
    pending: ['Remaining skill-retention checks are in progress.', 'Проверки сохранения навыков ещё выполняются.'],
    snapshot: ['RECORDED SNAPSHOT / ', 'СНИМОК РЕЗУЛЬТАТОВ / '],
    telemetry: [' · Not live telemetry', ' · Не живая телеметрия'],
    meshError: ['Mesh unavailable', 'Сетка недоступна'],
    dataError: ['Project snapshot unavailable', 'Снимок проекта недоступен'],
    webglError: ['WebGL is unavailable. Reference geometry is listed below.', 'WebGL недоступен. Описание эталонной геометрии приведено ниже.'],
    serverError: ['Run a local HTTP server to load the project snapshot.\nSee web/README.md.', 'Для загрузки снимка проекта запустите локальный HTTP-сервер.\nИнструкции: web/README.md.']
  };
  const attributes = {
    'Original technical documentation in English': 'Оригинальная техническая документация на английском',
    'cad-env home': 'Главная cad-env', 'Main navigation': 'Основная навигация', 'Toggle navigation': 'Открыть или закрыть меню',
    'Interactive native CAD reference viewer': 'Интерактивный просмотр нативных CAD-эталонов',
    'Interactive 3D CAD model. Drag to orbit; use the zoom buttons.': 'Интерактивная 3D-модель CAD. Перетаскивайте для вращения; кнопки меняют масштаб.',
    'Toggle wireframe': 'Переключить каркас', 'Zoom out': 'Уменьшить', 'Zoom in': 'Увеличить', 'Reset camera': 'Сбросить камеру',
    'Explore the agent loop': 'Изучить цикл агента',
    'Select example part': 'Выбрать пример детали'
  };
  const originals = new WeakMap();
  const t = (key, values = {}) => {
    const entry = ui[key];
    let value = entry ? entry[locale === 'ru' ? 1 : 0] : strings[key]?.[locale] ?? key;
    for (const [name, text] of Object.entries(values)) value = value.replaceAll('{'+name+'}', String(text));
    return value;
  };
  function apply() {
    document.documentElement.lang = locale;
    document.querySelectorAll('[data-number]').forEach(el => el.textContent = new Intl.NumberFormat(locale === 'ru' ? 'ru-RU' : 'en-GB').format(Number(el.dataset.number))); 
    document.querySelectorAll('[data-i18n]').forEach(el => { const value = strings[el.dataset.i18n]?.[locale]; if (value !== undefined) el.textContent = value; });
    document.querySelectorAll('[data-language]').forEach(el => el.setAttribute('aria-pressed', String(el.dataset.language === locale)));
    document.querySelectorAll('[aria-label],[title]').forEach(el => {
      let original = originals.get(el); if (!original) { original = {}; originals.set(el, original); }
      for (const name of ['aria-label', 'title']) {
        if (!el.hasAttribute(name)) continue;
        original[name] ??= el.getAttribute(name);
        if (attributes[original[name]]) el.setAttribute(name, locale === 'ru' ? attributes[original[name]] : original[name]);
      }
    });
    const guide = !!document.querySelector('.guide');
    document.title = t(guide ? 'guideTitle' : 'mainTitle');
    document.querySelector('meta[name="description"]')?.setAttribute('content', t(guide ? 'guideDescription' : 'mainDescription'));
    document.dispatchEvent(new CustomEvent('cad-languagechange'));
  }
  window.CAD_I18N = { t, get locale() { return locale; }, set(lang) {
    if (!['en', 'ru'].includes(lang)) return;
    locale = lang;
    try { localStorage.setItem('cad-env-language', locale); } catch { /* Preference remains in memory if storage is disabled. */ }
    const url = new URL(location.href); url.searchParams.set('lang', locale); history.replaceState(null, '', url);
    apply();
  }};
  document.querySelectorAll('[data-language]').forEach(button => button.addEventListener('click', () => window.CAD_I18N.set(button.dataset.language)));
  apply();
  fetch('translations.json').then(response => { if (!response.ok) throw new Error('Translation catalog unavailable'); return response.json(); }).then(catalog => { strings = catalog; apply(); }).catch(error => console.error(error));
})();
