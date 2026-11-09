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
    mainTitle: ['cad-env / Teach models to work in CAD.', 'cad-env / Учим модели работать в CAD.'],
    guideTitle: ['Getting started / cad-env', 'Начало работы / cad-env'],
    blogTitle: ['Project updates / cad-env', 'Обновления проекта / cad-env'],
    articleTitle: ['One part, two drawing sets / cad-env', 'Одна деталь — два комплекта чертежей / cad-env'],
    blogDescription: ['Updates to cad-env, drawing examples and notes from our CAD-agent experiments.', 'Обновления cad-env, примеры чертежей и заметки об обучении CAD-агентов.'],
    articleDescription: ['Paired ISO and GOST drawing profiles for CAD-agent training, with generated examples and PDF downloads.', 'Парные профили ISO и ГОСТ для обучения CAD-агентов: примеры чертежей и PDF для скачивания.'],
    mainDescription: ['CAD tools, drawing datasets and training scripts for language and vision models. Built around FreeCAD and CadQuery.', 'CAD-инструменты, чертежи и скрипты обучения для языковых и визуальных моделей. На основе FreeCAD и CadQuery.'],
    guideDescription: ['Install cad-env, generate a dataset and understand how CAD-agent experiments are evaluated.', 'Установка cad-env, генерация данных и проверка результатов CAD-агентов.'],
    copied: ['Copied to clipboard', 'Скопировано в буфер обмена'],
    clipboard: ['Clipboard unavailable. Select the code to copy.', 'Буфер обмена недоступен. Выделите код для копирования.'],
    bolt_flange: ['Bolt flange', 'Фланец'],
    counterbored_plate: ['Counterbored plate', 'Плита с цековками'],
    stepped_bore: ['Stepped bore', 'Ступенчатое отверстие'],
    nativeReference: ['FreeCAD · validated TRAIN reference', 'FreeCAD · проверенный эталон из TRAIN'],
    actionCount: ['{count} ACTIONS · VERIFIED REFERENCE', '{count} ДЕЙСТВИЙ · ПРОВЕРЕННЫЙ ЭТАЛОН'],
    sequential: ['TASKS · SEQUENTIAL GREEDY EVALUATION', 'ЗАДАЧ · ПОСЛЕДОВАТЕЛЬНЫЙ GREEDY-РЕЖИМ'],
    simple: ['Simple tasks', 'Базовые задачи'], complex: ['Complex builds', 'Сложные построения'],
    grounding: ['Visual selection', 'Визуальный выбор'], entities: ['Entity-labelled selection', 'Выбор с подписями объектов'],
    batch: ['Batch ≤4 · 8 episode workers', 'Пакет ≤4 · 8 параллельных сред'],
    complete: ['This earlier run is complete. Retention was checked in a separate batch evaluation.', 'Этот ранний прогон завершён. Сохранение навыков проверено отдельно в пакетном режиме.'],
    pending: ['Remaining skill-retention checks are in progress.', 'Проверки сохранения навыков ещё выполняются.'],
    snapshot: ['RECORDED SNAPSHOT / ', 'СНИМОК РЕЗУЛЬТАТОВ / '],
    telemetry: [' · Saved results, not live progress', ' · Сохранённые результаты, не текущий прогресс'],
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
    'Select example part': 'Выбрать пример детали',
    'Read about ISO and GOST drawings': 'Читать о чертежах ISO и ГОСТ',
    'Open full-size drawing': 'Открыть чертёж в полном размере',
    'ISO and GOST drawing comparison': 'Сравнение оформления ISO и ГОСТ',
    'bearing cartridge — ISO sheet': 'Корпус подшипника — обзорный лист ISO',
    'bearing cartridge — GOST sheet': 'Корпус подшипника — обзорный лист ГОСТ',
    'bearing cartridge — ISO table': 'Корпус подшипника — фрагмент таблицы размеров ISO',
    'adapter flange — ISO sheet': 'Переходный фланец — обзорный лист ISO',
    'alignment fixture — ISO sheet': 'Установочное приспособление — обзорный лист ISO'
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
    document.querySelectorAll('[aria-label],[title],img[alt]').forEach(el => {
      let original = originals.get(el); if (!original) { original = {}; originals.set(el, original); }
      for (const name of ['aria-label', 'title', 'alt']) {
        if (!el.hasAttribute(name)) continue;
        original[name] ??= el.getAttribute(name);
        if (attributes[original[name]]) el.setAttribute(name, locale === 'ru' ? attributes[original[name]] : original[name]);
      }
    });
    document.querySelectorAll('a[href]').forEach(link => {
      const value = link.getAttribute('href');
      if (!value || /^[a-z]+:/i.test(value) || !value.split(/[?#]/)[0].endsWith('.html')) return;
      const url = new URL(value, location.href);
      url.searchParams.set('lang', locale);
      link.setAttribute('href', value.split(/[?#]/)[0] + url.search + url.hash);
    });
    const page = document.body.dataset.page || (document.querySelector('.guide') ? 'guide' : 'main');
    document.title = t(page + 'Title');
    document.querySelector('meta[name="description"]')?.setAttribute('content', t(page + 'Description'));
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
