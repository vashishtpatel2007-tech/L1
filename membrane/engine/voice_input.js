/* Dictation writes a draft only. Running always requires Enter or Run agent. */
(() => {
  const input = document.getElementById('cmd');
  const run = document.getElementById('bAgent');
  const button = document.createElement('button');
  button.type = 'button'; button.id = 'dictate'; button.textContent = '🎙 Dictate';
  button.setAttribute('aria-label', 'Dictate command');
  button.setAttribute('aria-pressed', 'false');
  button.title = 'Speak a command, review it, then press Enter. Your browser may use an online speech service.';
  input.insertAdjacentElement('afterend', button);
  const status = document.createElement('p');
  status.className = 'sub'; status.id = 'dictationStatus';
  status.setAttribute('role', 'status'); status.setAttribute('aria-live', 'polite');
  input.closest('section').appendChild(status);
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  let recognition = null, active = false, heard = false, failed = false, pending = false, acceptResults = false;
  const submit = () => { if (input.value.trim() && !run.disabled) run.click(); };
  const stop = () => { if (active && recognition) recognition.stop(); };
  button.addEventListener('click', () => {
    if (active) { stop(); return; }
    if (!Recognition) {
      status.textContent = 'Voice input is unavailable in this browser. Open this page in Chrome or Edge, or type your command.';
      return;
    }
    if (run.disabled) { status.textContent = 'Wait for the current task to finish before dictating.'; return; }
    recognition = new Recognition(); recognition.lang = 'en-IN';
    recognition.continuous = true; recognition.interimResults = true; recognition.maxAlternatives = 1;
    active = true; heard = false; failed = false; pending = false; acceptResults = true;
    button.textContent = '■ Stop dictation'; button.setAttribute('aria-pressed', 'true');
    status.textContent = 'Allow microphone access, then speak. Review the text and press Enter to run.';
    recognition.onstart = () => { status.textContent = 'Listening… speak your command. Stop to review, or press Enter to run.'; };
    recognition.onresult = event => {
      if (!acceptResults) return;
      const text = Array.from(event.results, result => result[0].transcript).join(' ').trim();
      if (text) { heard = true; input.value = text; }
    };
    recognition.onerror = event => {
      failed = true; pending = false;
      const messages = {'not-allowed':'Microphone permission was denied. Allow it in browser settings or type your command.',
        'audio-capture':'No microphone is available. Connect one or type your command.',
        'network':'Speech recognition could not connect. Your draft is preserved; type or try again.',
        'no-speech':'No speech was detected. Try again or type your command.',
        'language-not-supported':'This speech service does not support the selected language. Type your command.'};
      status.textContent = messages[event.error] || 'Dictation stopped. Your draft is preserved; try again or type.';
    };
    recognition.onend = () => {
      active = false; acceptResults = false;
      button.textContent = '🎙 Dictate'; button.setAttribute('aria-pressed', 'false');
      if (!failed) status.textContent = heard ? 'Review your command. Press Enter to run.' : 'No speech captured. Your previous text is unchanged.';
      input.focus();
      if (pending && heard && !failed) submit();
      pending = false;
    };
    try { recognition.start(); } catch (error) {
      active = false; acceptResults = false; button.textContent = '🎙 Dictate'; button.setAttribute('aria-pressed', 'false');
      status.textContent = 'Microphone could not start. Type your command or try again.';
    }
  });
  input.addEventListener('keydown', event => {
    if (event.key !== 'Enter' || event.isComposing || event.repeat) return;
    event.preventDefault();
    if (active) { pending = true; stop(); } else submit();
  });
  input.addEventListener('input', () => { if (active) { acceptResults = false; pending = false; stop(); } });
  run.addEventListener('click', () => { acceptResults = false; pending = false; stop(); }, true);
  window.addEventListener('pagehide', () => { acceptResults = false; pending = false; if (recognition) recognition.abort(); });
})();
