'use strict';

const videos = [...document.querySelectorAll('video')];
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
const manuallyPaused = new WeakSet();
const observerPaused = new WeakSet();
videos.forEach(video => {
  video.addEventListener('play', () => manuallyPaused.delete(video));
  video.addEventListener('pause', () => {
    if (observerPaused.has(video)) observerPaused.delete(video);
    else manuallyPaused.add(video);
  });
});
function pauseForVisibility(video) {
  if (!video.paused) { observerPaused.add(video); video.pause(); }
}
// Playback follows real recorded timestamps. Autoplay only when visible,
// muted, and permitted; native controls remain available at all times.
const videoObserver = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    const video = entry.target;
    if (entry.isIntersecting && entry.intersectionRatio >= 0.45 && !document.hidden && !reducedMotion.matches && !manuallyPaused.has(video)) {
      video.play().catch(() => {});
    } else if (!entry.isIntersecting || entry.intersectionRatio < 0.1) {
      pauseForVisibility(video);
    }
  });
}, {threshold: [0, 0.1, 0.45]});
videos.forEach(video => videoObserver.observe(video));
document.addEventListener('visibilitychange', () => { if (document.hidden) videos.forEach(pauseForVisibility); });
reducedMotion.addEventListener('change', () => { if (reducedMotion.matches) videos.forEach(pauseForVisibility); });

const citationButton = document.querySelector('#copy-citation');
citationButton.addEventListener('click', async () => {
  const citation = document.querySelector('#bibtex').textContent;
  let copied = false;
  try {
    await navigator.clipboard.writeText(citation);
    copied = true;
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(document.querySelector('#bibtex'));
    selection.removeAllRanges(); selection.addRange(range);
    try { copied = document.execCommand('copy'); } catch { /* Leave selected for manual copy. */ }
    if (copied) selection.removeAllRanges();
  }
  citationButton.textContent = copied ? 'Copied ✓' : 'Select & copy below';
  document.querySelector('#copy-status').textContent = copied ? 'BibTeX copied to clipboard.' : 'Citation selected. Use your copy shortcut.';
  setTimeout(() => { citationButton.textContent = 'Copy BibTeX'; }, 2500);
});

const dialog = document.querySelector('#figure-dialog');
document.querySelectorAll('[data-zoom]').forEach(button => button.addEventListener('click', () => {
  const image = dialog.querySelector('img');
  image.src = button.dataset.zoom;
  image.alt = button.querySelector('img').alt;
  dialog.setAttribute('aria-label', button.getAttribute('aria-label'));
  dialog.showModal();
}));
dialog.querySelector('.dialog-close').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });

const navLinks = [...document.querySelectorAll('[data-section-link]')];
const sectionObserver = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) navLinks.forEach(link => {
      if (link.hash === '#' + entry.target.id) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
  });
}, {rootMargin: '-15% 0px -65% 0px'});
[...new Set(navLinks.map(link => link.hash))].forEach(hash => { const section = document.querySelector(hash); if (section) sectionObserver.observe(section); });
