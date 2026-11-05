/* Move existing live panels; never duplicate streams or simulation state. */
(() => {
  const views = ['pov', 'map', 'cinematic'].map(id => document.getElementById(id).closest('section'));
  const logs = document.getElementById('alog').closest('section');
  const overlay = document.createElement('section');
  overlay.id = 'allViews'; overlay.hidden = true;
  overlay.setAttribute('aria-label', 'All live simulation views');
  const toolbar = document.createElement('div'); toolbar.className = 'all-views-toolbar';
  const title = document.createElement('strong'); title.textContent = 'All live views';
  const exit = document.createElement('button'); exit.textContent = '↙ Back to dashboard';
  toolbar.append(title, exit); overlay.append(toolbar); document.body.append(overlay);
  const button = document.createElement('button'); button.id = 'allViewsButton';
  button.textContent = '⛶ All views fullscreen'; button.setAttribute('aria-expanded', 'false');
  document.querySelector('header').append(button);
  let active = false, native = false, saved = [], previousFocus = null, opening = false;
  function restore() {
    if (!active) return;
    active = false; native = false;
    for (const [panel, marker] of saved) marker.replaceWith(panel);
    saved = []; overlay.hidden = true; document.body.classList.remove('all-views-open');
    button.setAttribute('aria-expanded', 'false'); previousFocus?.focus();
  }
  async function close() {
    const leaveNative = native && document.fullscreenElement === document.documentElement;
    restore();
    if (leaveNative) await document.exitFullscreen().catch(() => {});
  }
  button.addEventListener('click', async () => {
    if (active || opening) return;
    opening = true;
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      previousFocus = document.activeElement;
      for (const panel of [...views, logs]) {
        panel.classList.remove('expanded');
        const marker = document.createComment('live panel position');
        panel.before(marker); saved.push([panel, marker]); overlay.append(panel);
      }
      active = true; overlay.hidden = false; document.body.classList.add('all-views-open');
      button.setAttribute('aria-expanded', 'true'); exit.focus();
      try {
        await document.documentElement.requestFullscreen(); native = active;
        if (!active && document.fullscreenElement === document.documentElement) await document.exitFullscreen();
      }
      catch (_) { native = false; } // viewport fallback retains the same three columns
    } finally { opening = false; }
  });
  exit.addEventListener('click', close);
  document.addEventListener('fullscreenchange', () => {
    if (active && native && !document.fullscreenElement) restore();
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && active) { event.preventDefault(); close(); }
  });

  const dialog = document.createElement('dialog'); dialog.id = 'selectionIssue';
  dialog.setAttribute('aria-labelledby', 'selectionTitle');
  const heading = document.createElement('h2'); heading.id = 'selectionTitle';
  heading.textContent = 'Please be more specific';
  const message = document.createElement('p'), choices = document.createElement('ul');
  const edit = document.createElement('button'); edit.textContent = 'Edit command';
  dialog.append(heading, message, choices, edit); document.body.append(dialog);
  edit.addEventListener('click', async () => {
    dialog.close(); if (active) await close();
    document.getElementById('cmd').focus();
  });
  window.showSelectionIssue = issue => {
    if (!issue) return;
    heading.textContent = issue.code === 'ambiguous' ? 'Multiple matches — please be more specific' : 'Object could not be identified safely';
    message.textContent = issue.message; choices.replaceChildren();
    for (const text of issue.candidates || []) {
      const item = document.createElement('li'); item.textContent = text; choices.append(item);
    }
    if (!dialog.open) dialog.showModal();
  };
})();
