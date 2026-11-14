// Shared shell behavior for the patient app (app.html) and the doctor page:
// the sliding tab indicator and the entrance animation of a newly shown view.

const REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)');

// Moves the indicator pill under the tab marked aria-current="page". The first
// placement does not animate, so the page never opens with a slide.
export function syncTabs(nav) {
  if (!nav) return;
  let indicator = nav.querySelector('.tab-indicator');
  if (!indicator) {
    indicator = document.createElement('span');
    indicator.className = 'tab-indicator';
    indicator.setAttribute('aria-hidden', 'true');
    nav.prepend(indicator);
    const resize = new ResizeObserver(() => place(nav, indicator, false));
    resize.observe(nav);
    nav.querySelectorAll('.tab').forEach(tab => resize.observe(tab));
  }
  place(nav, indicator, nav.dataset.placed === 'true');
}

function place(nav, indicator, animate) {
  const active = nav.querySelector('.tab[aria-current="page"]');
  if (!active) {
    indicator.style.opacity = '0';
    return;
  }
  if (!animate || REDUCED.matches) indicator.classList.add('no-anim');
  indicator.style.opacity = '1';
  indicator.style.width = active.offsetWidth + 'px';
  indicator.style.height = active.offsetHeight + 'px';
  indicator.style.transform = `translate(${active.offsetLeft}px, ${active.offsetTop}px)`;
  if (indicator.classList.contains('no-anim')) {
    void indicator.offsetWidth;  // apply the jump before turning transitions back on
    indicator.classList.remove('no-anim');
  }
  nav.dataset.placed = 'true';
}

// Plays the staggered entrance on a view that was just switched to. Re-renders
// of the same view (a checkbox, a new message) do not replay it.
export function enter(el) {
  if (!el || REDUCED.matches) return;
  el.classList.remove('is-entering');
  void el.offsetWidth;
  el.classList.add('is-entering');
  clearTimeout(el._enterTimer);
  el._enterTimer = setTimeout(() => el.classList.remove('is-entering'), 1200);
}

// "Dr. Ting Kai" -> "TK", "demo2" -> "D". Titles such as "Dr." are skipped.
export function initials(name) {
  const words = String(name || '').replace(/^(dr|mr|mrs|ms|prof)\.?\s+/i, '').trim().split(/\s+/).filter(Boolean);
  if (!words.length) return '?';
  const letters = words.length === 1 ? words[0].charAt(0) : words[0].charAt(0) + words[words.length - 1].charAt(0);
  return letters.toUpperCase();
}
