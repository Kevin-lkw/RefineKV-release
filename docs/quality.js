'use strict';

(() => {
  const buttons = [...document.querySelectorAll('[data-quality-model]')];
  const panels = [...document.querySelectorAll('[data-quality-panel]')];
  buttons.forEach(button => button.addEventListener('click', () => {
    const model = button.dataset.qualityModel;
    buttons.forEach(item => item.setAttribute('aria-pressed', String(item === button)));
    panels.forEach(panel => { panel.hidden = panel.dataset.qualityPanel !== model; });
    document.getElementById('quality-status').textContent = `Showing ${button.textContent} results for LongBench, RULER, and HELMET.`;
  }));
})();
