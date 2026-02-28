/**
 * SwasthyaSetu Medical AI - Application JavaScript
 * 
 * Handles diagnosis form submission, report rendering, PDF generation, and visual aid display.
 * Includes security sanitization for XSS prevention and rate limiting handling.
 * 
 * @version 1.1.0
 * @license MIT
 */

document.addEventListener('DOMContentLoaded', function () {
  // DOM Element References
  const form = document.getElementById('diagnosis-form');
  const symptomsInput = document.getElementById('symptoms');
  const locationInput = document.getElementById('location');
  const learnModeInput = document.getElementById('learn-mode');
  const loading = document.getElementById('loading');
  const reportSection = document.getElementById('report-section');
  const reportContent = document.getElementById('report-content');
  const reportError = document.getElementById('report-error');
  const rateLimitAlert = document.getElementById('rate-limit-alert');
  const exampleBtn = document.getElementById('example-btn');
  const clearBtn = document.getElementById('clear-btn');
  const exampleChips = document.querySelectorAll('.example-chip');
  const progressIndicator = document.getElementById('progress-indicator');
  const progressBar = document.getElementById('progress-bar');
  const progressStep = document.getElementById('progress-step');
  const downloadBtn = document.getElementById('download-btn');
  const printBtn = document.getElementById('print-btn');
  const toggleAdvancedBtn = document.getElementById('toggle-advanced-btn');
  const advancedSection = document.getElementById('advanced-section');
  const advancedContent = document.getElementById('advanced-content');
  // New UI element refs
  const charCounter = document.getElementById('char-counter');
  const symptomsError = document.getElementById('symptoms-error');
  const locationError = document.getElementById('location-error');
  const severityInput = document.getElementById('severity');
  const severityValue = document.getElementById('severity-value');
  const shareBtn = document.getElementById('share-btn');
  const tryAnotherBtn = document.getElementById('try-another-btn');
  const tryAnotherSection = document.getElementById('try-another-section');
  const emergencyBanner = document.getElementById('emergency-banner');
  const copyToast = document.getElementById('copy-toast');
  const skeletonCards = document.getElementById('skeleton-cards');

  // Stores the most recent query for the symptom-echo band in the report
  let lastQuery = { symptoms: '', location: '' };
  // Stores the latest full report for the share button
  let currentReport = null;

  /**
   * Toggle advanced section visibility
   */
  if (toggleAdvancedBtn && advancedSection) {
    toggleAdvancedBtn.addEventListener('click', function () {
      if (advancedSection.classList.contains('hidden')) {
        advancedSection.classList.remove('hidden');
        toggleAdvancedBtn.textContent = 'Hide Advanced Details';
      } else {
        advancedSection.classList.add('hidden');
        toggleAdvancedBtn.textContent = 'Show Advanced Details';
      }
    });
  }

  // ── Char counter ──────────────────────────────────────────────────────────
  function updateCharCounter() {
    if (!charCounter || !symptomsInput) return;
    const len = symptomsInput.value.length;
    charCounter.textContent = `${len} / 2000`;
    charCounter.className = 'char-counter' + (len > 1900 ? ' over' : len > 1500 ? ' warn' : '');
  }
  if (symptomsInput) symptomsInput.addEventListener('input', updateCharCounter);

  // ── Severity slider ───────────────────────────────────────────────────────
  if (severityInput && severityValue) {
    severityInput.addEventListener('input', function () {
      severityValue.textContent = this.value;
      const pct = ((this.value - 1) / 9 * 100).toFixed(1);
      this.style.background = `linear-gradient(to right, #138808 ${pct}%, #e2e8f0 ${pct}%)`;
    });
  }

  // ── Ctrl+Enter submits the form ───────────────────────────────────────────
  if (symptomsInput) {
    symptomsInput.addEventListener('keydown', function (e) {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
      }
    });
  }

  // ── Restore learn-mode toggle visual state on page load ───────────────────
  (function () {
    const tb = document.getElementById('toggle-bg');
    const td = document.getElementById('toggle-dot');
    const lm = document.getElementById('learn-mode');
    if (lm && tb && td && lm.checked) {
      tb.className = 'block w-12 h-7 rounded-full transition-colors duration-300 bg-[#138808]';
      td.style.transform = 'translateX(20px)';
    }
  }());

  // ── Clipboard toast helper ────────────────────────────────────────────────
  function showCopyToast(msg) {
    if (!copyToast) return;
    copyToast.textContent = msg || '\u2705 Copied!';
    copyToast.classList.add('show');
    setTimeout(() => copyToast.classList.remove('show'), 2200);
  }

  // ── Confidence history (localStorage) ────────────────────────────────────
  function saveConfidence(diagName, conf) {
    try {
      const raw = localStorage.getItem('ss_conf_history');
      const h = raw ? JSON.parse(raw) : {};
      if (!h[diagName]) h[diagName] = [];
      h[diagName] = [...h[diagName].slice(-9), conf];
      localStorage.setItem('ss_conf_history', JSON.stringify(h));
    } catch (_) { }
  }
  function getConfidenceHistory(diagName) {
    try {
      const raw = localStorage.getItem('ss_conf_history');
      return (raw ? JSON.parse(raw) : {})[diagName] || [];
    } catch (_) { return []; }
  }
  function renderConfidenceChart(diagName) {
    const hist = getConfidenceHistory(diagName);
    if (hist.length < 2) return '';
    const avg = hist.reduce((a, b) => a + b, 0) / hist.length;
    const bars = hist.map((v, i) => {
      const h = Math.max(4, Math.round(v * 36));
      const isCur = i === hist.length - 1;
      return `<div class="conf-bar${isCur ? ' current' : ''}" style="height:${h}px" title="${(v * 100).toFixed(0)}%"></div>`;
    }).join('');
    return `
      <div class="mt-4 p-3 bg-slate-800/30 rounded-lg border border-slate-700/40">
        <div class="text-xs font-bold text-indigo-400 mb-2 uppercase tracking-wider">Confidence History</div>
        <div class="conf-chart">${bars}</div>
        <div class="text-xs text-slate-400 mt-1">Historical avg: <strong class="text-indigo-300">${(avg * 100).toFixed(0)}%</strong> across ${hist.length} run${hist.length === 1 ? '' : 's'}</div>
      </div>`;
  }

  // Progress steps for UI feedback
  const steps = [
    'Starting...',
    'Analyzing symptoms',
    'Retrieving medical context',
    'Generating diagnosis',
    'Validating diagnosis',
    'Patient education',
    'Checking for bias',
    'Formatting report',
    'Done!'
  ];

  /**
   * Update progress indicator
   * @param {number} stepIdx - Index of current step
   */
  function setProgress(stepIdx) {
    progressIndicator.classList.remove('hidden');
    const percent = Math.round((stepIdx / (steps.length - 1)) * 100);
    progressBar.style.width = percent + '%';
    progressStep.textContent = steps[stepIdx] || '';
  }

  /**
   * Reset progress indicator to hidden state
   */
  function resetProgress() {
    progressIndicator.classList.add('hidden');
    progressBar.style.width = '0%';
    progressStep.textContent = '';
  }

  /**
   * Escape HTML special characters to prevent XSS attacks
   * @param {string} input - Raw input string
   * @returns {string} Escaped HTML string
   */
  function escapeHtml(input) {
    if (input === null || input === undefined) return '';
    return String(input)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  /**
   * Sanitize URL to prevent javascript: protocol injection
   * @param {string} url - URL to sanitize
   * @returns {string} Safe URL or '#' if unsafe
   */
  function sanitizeUrl(url) {
    try {
      const parsed = new URL(String(url), window.location.origin);
      if (parsed.protocol === 'http:' || parsed.protocol === 'https:') return parsed.href;
    } catch (_) { }
    return '#';
  }

  /**
   * Recursively sanitize object values
   * @param {*} value - Value to sanitize
   * @returns {*} Sanitized value
   */
  function sanitizeDeep(value) {
    if (typeof value === 'string') return escapeHtml(value);
    if (Array.isArray(value)) return value.map(sanitizeDeep);
    if (value && typeof value === 'object') {
      const out = {};
      Object.keys(value).forEach((k) => {
        out[k] = sanitizeDeep(value[k]);
      });
      return out;
    }
    return value;
  }

  /**
   * Sanitize entire report object
   * @param {Object} report - Raw report from API
   * @returns {Object} Sanitized report
   */
  function sanitizeReport(report) {
    const safe = sanitizeDeep(report || {});
    if (safe?.routing?.results && Array.isArray(safe.routing.results)) {
      safe.routing.results = safe.routing.results.map((item) => ({
        title: item?.title || '',
        url: sanitizeUrl(item?.url),
      }));
    }
    return safe;
  }

  /**
   * Render visual aid image with loading state and error handling
   * @param {string} imageUrl - URL of the image to display
   * @param {string} diagnosisName - Name of the diagnosis for alt text
   * @returns {string} HTML string for the visual aid component
   */
  function renderVisualAid(imageUrl, diagnosisName) {
    if (!imageUrl) {
      return `
        <div class="visual-aid-container">
          <div class="visual-aid-error">
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"/>
            </svg>
            <p>Visual aid not available for this condition</p>
          </div>
        </div>
      `;
    }

    const safeAltText = escapeHtml(`Medical illustration of ${diagnosisName || 'condition'}`);
    const safeUrl = sanitizeUrl(imageUrl);
    // Use a per-call unique suffix so IDs don't collide if renderVisualAid is ever called
    // more than once on the same page.
    const uid = Math.random().toString(36).slice(2, 8);
    const loadingId = `va-loading-${uid}`;
    const imageId = `va-img-${uid}`;
    const errorId = `va-err-${uid}`;

    return `
      <div class="visual-aid-container">
        <div class="visual-aid-loading" id="${loadingId}">
          <div class="spinner"></div>
          <p style="color: #64748b; font-size: 12px;">Generating medical illustration...</p>
        </div>
        <img 
          src="${safeUrl}" 
          alt="${safeAltText}"
          class="visual-aid-image"
          id="${imageId}"
          style="display: none;"
          onload="
            var l=document.getElementById('${loadingId}'); if(l) l.style.display='none';
            var i=document.getElementById('${imageId}'); if(i) i.style.display='block';
          "
          onerror="
            var l=document.getElementById('${loadingId}'); if(l) l.style.display='none';
            var e=document.getElementById('${errorId}'); if(e) e.style.display='flex';
          "
        />
        <div class="visual-aid-error" id="${errorId}" style="display: none;">
          <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
          </svg>
          <p>Unable to load visual aid</p>
        </div>
      </div>
      <div class="visual-aid-caption">
        <strong>AI-Generated Illustration:</strong> Educational visualization of ${escapeHtml(diagnosisName || 'this condition')}.
        <br>Not a diagnostic image. For educational purposes only.
      </div>
    `;
  }

  /**
   * Form submission handler
   */
  form.addEventListener('submit', async function (e) {
    e.preventDefault();

    // ─ Input validation with visible error messages ───────────────────────
    const symptoms = symptomsInput.value.trim();
    const location = locationInput.value.trim();
    const learn_mode = learnModeInput && learnModeInput.checked;
    let valid = true;

    if (!symptoms || symptoms.length < 5) {
      if (symptomsError) symptomsError.classList.remove('hidden');
      symptomsInput.setAttribute('aria-invalid', 'true');
      symptomsInput.focus();
      valid = false;
    } else {
      if (symptomsError) symptomsError.classList.add('hidden');
      symptomsInput.setAttribute('aria-invalid', 'false');
    }

    if (!location) {
      if (locationError) locationError.classList.remove('hidden');
      locationInput.setAttribute('aria-invalid', 'true');
      if (valid) locationInput.focus();
      valid = false;
    } else {
      if (locationError) locationError.classList.add('hidden');
      locationInput.setAttribute('aria-invalid', 'false');
    }

    if (!valid) return;

    // ─ Prepend severity level to symptoms if set away from neutral (5) ───
    const sevVal = parseInt(severityInput?.value || '5');
    let sevLabel = sevVal <= 3 ? 'mild' : sevVal <= 6 ? 'moderate' : sevVal <= 8 ? 'severe' : 'extreme';
    const fullSymptoms = sevVal !== 5
      ? `Symptom severity: ${sevVal}/10 (${sevLabel}). ${symptoms}`
      : symptoms;

    // ─ Save for report echo ───────────────────────────────────────────
    lastQuery = { symptoms, location };

    // ─ Reset report state ────────────────────────────────────────────
    loading.classList.remove('hidden');
    resetProgress();
    setProgress(1);
    reportSection.classList.add('hidden');
    reportError.classList.add('hidden');
    if (rateLimitAlert) rateLimitAlert.classList.add('hidden');
    if (emergencyBanner) emergencyBanner.classList.add('hidden');
    if (tryAnotherSection) tryAnotherSection.classList.add('hidden');
    reportContent.innerHTML = '';

    // ─ Smooth progress timer ────────────────────────────────────────
    let currentStep = 1;
    const progressTimer = setInterval(() => {
      if (currentStep < 7) { currentStep++; setProgress(currentStep); }
    }, 2200);

    try {
      const response = await fetch('/diagnose', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symptoms: fullSymptoms, location, learn_mode })
      });
      clearInterval(progressTimer);
      setProgress(7);
      let data = {};
      try { data = await response.json(); } catch (_) { data = {}; }

      if (response.status === 429) {
        showRateLimit(data);
      } else if (!response.ok) {
        showError(data.error || `Request failed (${response.status}).`);
      } else if (data.status === 'success') {
        currentReport = data.report;
        renderReport(data.report);
        setProgress(8);
        reportSection.classList.remove('hidden');
        setTimeout(() => reportSection.scrollIntoView({ behavior: 'smooth', block: 'start' }), 120);
      } else {
        showError(data.error || 'Unknown error.');
      }
    } catch (err) {
      clearInterval(progressTimer);
      showError(err.message);
    } finally {
      loading.classList.add('hidden');
      setTimeout(resetProgress, 1200);
    }
  });

  /**
   * Display error message
   * @param {string} msg - Error message to display
   */
  function showError(msg) {
    if (rateLimitAlert) rateLimitAlert.classList.add('hidden');
    reportError.textContent = 'Error: ' + msg;
    reportError.classList.remove('hidden');
    reportSection.classList.remove('hidden');
  }

  /**
   * Display rate limit information
   * @param {Object} data - Rate limit data from API
   */
  function showRateLimit(data) {
    if (!rateLimitAlert) {
      showError('Rate limit reached. Please retry later.');
      return;
    }
    const retryAfter = Number(data?.retry_after_seconds || 0);
    const minuteLimit = Number(data?.limits?.minute || 0);
    const hourLimit = Number(data?.limits?.hour || 0);
    const waitText = retryAfter > 0
      ? `Please wait ${retryAfter} second${retryAfter === 1 ? '' : 's'} before retrying.`
      : 'Please wait a short while before retrying.';
    const limitText = (minuteLimit || hourLimit)
      ? `Hard limits: ${minuteLimit || '-'} requests/min and ${hourLimit || '-'} requests/hour.`
      : '';
    rateLimitAlert.textContent = `Rate limit reached. ${waitText} ${limitText}`.trim();
    rateLimitAlert.classList.remove('hidden');
    reportError.classList.add('hidden');
    reportSection.classList.remove('hidden');
  }

  // Sample case button
  exampleBtn.addEventListener('click', function () {
    symptomsInput.value = 'Patient presents with high fever, chills, headache, and fatigue. Recently returned from a trip to a tropical region known for mosquito-borne illnesses.';
    symptomsInput.focus();
    updateCharCounter();
  });

  // Clear input button
  clearBtn.addEventListener('click', function () {
    symptomsInput.value = '';
    locationInput.value = '';
    updateCharCounter();
    if (symptomsError) symptomsError.classList.add('hidden');
    if (locationError) locationError.classList.add('hidden');
    symptomsInput.focus();
    reportSection.classList.add('hidden');
    reportContent.innerHTML = '';
    reportError.classList.add('hidden');
    if (rateLimitAlert) rateLimitAlert.classList.add('hidden');
    if (tryAnotherSection) tryAnotherSection.classList.add('hidden');
    if (emergencyBanner) emergencyBanner.classList.add('hidden');
  });

  // Example chips click handlers
  exampleChips.forEach(chip => {
    chip.addEventListener('click', function () {
      symptomsInput.value = chip.textContent;
      symptomsInput.focus();
    });
    chip.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') {
        symptomsInput.value = chip.textContent;
        symptomsInput.focus();
      }
    });
  });

  /**
   * Render the diagnosis report
   * @param {Object} report - Report data from API
   */
  function renderReport(report) {
    // Save raw confidence before sanitization (for localStorage history)
    const _rawPrimary = String(report?.diagnosis?.primary || '');
    const _rawConf = parseFloat(report?.diagnosis?.confidence || 0);
    if (_rawPrimary && _rawPrimary !== 'N/A') saveConfidence(_rawPrimary, _rawConf);

    report = sanitizeReport(report);
    if (!report || !report.diagnosis) { showError('No report data.'); return; }

    report.diagnosis = report.diagnosis || {};
    report.triage = report.triage || {};
    report.routing = report.routing || {};
    report.education = report.education || {};
    report.equity_check = report.equity_check || {};
    report.debug_info = report.debug_info || {};
    report.reasoning = Array.isArray(report.reasoning) ? report.reasoning : [];
    report.guidelines = Array.isArray(report.guidelines) ? report.guidelines : [];

    // ── Derived values ────────────────────────────────────────────────
    const confPct = (parseFloat(report.diagnosis.confidence || 0) * 100).toFixed(1);
    const confNum = parseFloat(confPct);
    const confColor = confNum >= 80
      ? 'bg-green-100 text-[#138808] border-green-200'
      : confNum >= 50
        ? 'bg-amber-100 text-amber-700 border-amber-200'
        : 'bg-red-100 text-red-600 border-red-200';

    const TRIAGE_LABELS = {
      self_care: 'Self Care at Home',
      clinic_visit: 'Visit a Clinic',
      urgent_care: 'Urgent Care Needed',
      emergency: '\u26a0 Emergency',
    };
    const triageLevelRaw = String(report.triage.level || '').toLowerCase();
    const triageLevelDisplay = TRIAGE_LABELS[triageLevelRaw] || String(report.triage.level || 'N/A');
    const isEmergency = triageLevelRaw.includes('emergency');
    const isUrgent = triageLevelRaw.includes('urgent');

    // Show / hide emergency banner
    if (emergencyBanner) {
      isEmergency ? emergencyBanner.classList.remove('hidden') : emergencyBanner.classList.add('hidden');
    }

    // Triage card colours
    let levelColor = 'text-[#138808]', bgColor = 'bg-green-50/50',
      iconColor = 'text-[#138808]', borderColor = 'border-green-100';
    if (isEmergency) {
      levelColor = 'text-red-600'; bgColor = 'bg-red-50/50';
      iconColor = 'text-red-500'; borderColor = 'border-red-200';
    } else if (isUrgent) {
      levelColor = 'text-[#FF9933]'; bgColor = 'bg-orange-50/50';
      iconColor = 'text-[#FF9933]'; borderColor = 'border-orange-200';
    }

    let html = '';

    // ── Symptom echo ─────────────────────────────────────────────────
    if (lastQuery.symptoms) {
      html += `
      <div class="bg-slate-50 border border-slate-200 rounded-xl p-4 flex items-start gap-3">
        <svg class="w-5 h-5 text-slate-400 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"/></svg>
        <div class="flex-1 min-w-0">
          <div class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Your Query</div>
          <div class="text-slate-700 font-medium text-sm line-clamp-3">${escapeHtml(lastQuery.symptoms)}</div>
          <div class="text-xs text-slate-400 mt-1">📍 ${escapeHtml(lastQuery.location)}</div>
        </div>
      </div>`;
    }

    // ── Diagnosis card ───────────────────────────────────────────────
    html += `
    <div class="bg-white rounded-2xl p-6 md:p-8 shadow-sm border border-slate-100 mb-6 relative overflow-hidden group hover:shadow-md transition-shadow">
       <div class="absolute top-0 right-0 w-32 h-32 bg-blue-50 rounded-full mix-blend-multiply filter blur-2xl opacity-70 group-hover:bg-blue-100 transition-colors"></div>
       <div class="relative z-10">
         <div class="flex items-center gap-3 mb-4">
           <div class="bg-blue-100 p-2 rounded-xl text-[#000080]">
             <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"></path></svg>
           </div>
           <h3 class="text-xl font-bold text-slate-800">Primary AI Diagnosis</h3>
         </div>
         <div class="flex flex-col sm:flex-row sm:items-baseline gap-2 mb-4">
           <span class="text-3xl sm:text-4xl font-extrabold text-[#000080]">${report.diagnosis.primary}</span>
           <span class="inline-flex items-center gap-1 ${confColor} text-sm font-bold px-3 py-1 rounded-full border">
             <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path></svg>
             Confidence: ${confPct}%
           </span>
           <button class="copy-diagnosis-btn ml-auto sm:ml-2 p-1.5 text-slate-400 hover:text-[#000080] rounded-lg hover:bg-slate-100 transition-all" data-copy="${report.diagnosis.primary} — Confidence: ${confPct}%" title="Copy diagnosis to clipboard">
             <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"/></svg>
           </button>
         </div>
    `;
    if (report.diagnosis.alternatives && report.diagnosis.alternatives.length) {
      html += `
         <div class="mt-4">
           <span class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2 block">Differential Diagnoses</span>
           <div class="flex flex-wrap gap-2">
             ${report.diagnosis.alternatives.map(alt => `<span class="bg-slate-50 text-slate-700 border border-slate-200 px-3 py-1 rounded-lg text-sm font-semibold">${alt}</span>`).join('')}
           </div>
         </div>
      `;
    }
    html += `
         <div class="mt-5 pt-4 border-t border-slate-100 flex items-center justify-between">
           <span class="text-sm font-semibold text-slate-500">Cross-Validation Status</span>
           <span class="text-sm font-bold ${String(report.diagnosis.validation_status).includes('Validated') ? 'text-[#138808]' : 'text-[#FF9933]'}">${report.diagnosis.validation_status}</span>
         </div>
       </div>
    </div>`;

    // ── Triage card ──────────────────────────────────────────────────
    if (report.triage) {
      html += `
      <div class="${bgColor} rounded-2xl p-6 md:p-8 shadow-sm border ${borderColor} mb-6">
        <div class="flex items-center gap-3 mb-4">
          <div class="bg-white p-2 rounded-xl shadow-sm ${iconColor}">
            <svg class="w-6 h-6 ${isEmergency ? 'animate-pulse' : ''}" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="${isEmergency ? 'M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z' : 'M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z'}"></path>
            </svg>
          </div>
          <h3 class="text-xl font-bold text-slate-800">Triage Assessment</h3>
        </div>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <div class="text-xs font-bold text-slate-500 uppercase tracking-wider mb-1">Urgency Level</div>
            <div class="text-2xl font-extrabold ${levelColor}">${triageLevelDisplay}</div>
          </div>
          <div>
            <div class="text-xs font-bold text-slate-500 uppercase tracking-wider mb-1">Recommended Action</div>
            <div class="text-lg font-bold text-slate-700">${report.triage.next_step}</div>
          </div>
        </div>
        <div class="mt-4 pt-4 border-t border-black/5">
          <p class="text-slate-600 font-medium leading-relaxed">${report.triage.explanation}</p>
        </div>
      </div>`;
    }

    // ── Routing card ─────────────────────────────────────────────────
    if (report.routing) {
      html += `
      <div class="bg-white rounded-2xl p-6 md:p-8 shadow-sm border border-slate-100 mb-6">
        <div class="flex items-center gap-3 mb-5">
          <div class="bg-indigo-50 p-2 rounded-xl text-indigo-600">
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4"></path></svg>
          </div>
          <h3 class="text-xl font-bold text-slate-800">Local Healthcare Providers</h3>
        </div>`;
      if (report.routing.results && report.routing.results.length) {
        html += `<div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">`;
        report.routing.results.forEach(r => {
          html += `
            <a href="${r.url}" target="_blank" rel="noopener noreferrer" class="block bg-slate-50 hover:bg-indigo-50 border border-slate-200 hover:border-indigo-200 p-4 rounded-xl transition-all group shadow-sm hover:shadow">
              <h4 class="font-bold text-slate-800 group-hover:text-indigo-700 line-clamp-2 mb-2 text-sm">${r.title}</h4>
              <div class="flex items-center text-xs font-semibold text-indigo-600">View Search <svg class="w-3 h-3 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"></path></svg></div>
            </a>`;
        });
        html += `</div>`;
      } else {
        html += `
          <div class="bg-slate-50 border border-dashed border-slate-300 p-6 rounded-xl text-center">
            <svg class="w-8 h-8 text-slate-400 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"/></svg>
            <p class="text-slate-500 font-semibold text-sm">Add your city above and re-run to find nearby specialists.</p>
          </div>`;
      }
      html += `</div>`;
    }

    // ── Education card ──────────────────────────────────────────────
    html += `
    <div class="bg-white rounded-2xl p-6 md:p-8 shadow-sm border border-slate-100 mb-6">
      <div class="flex items-center gap-3 mb-5">
        <div class="bg-teal-50 p-2 rounded-xl text-teal-600">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"></path></svg>
        </div>
        <h3 class="text-xl font-bold text-slate-800">Understanding Your Condition</h3>
      </div>
      <div class="space-y-6">
        <div>
          <h4 class="text-sm font-bold text-slate-800 mb-2">What is this?</h4>
          <p class="text-slate-600 font-medium leading-relaxed bg-slate-50 p-4 rounded-xl border border-slate-100">${report.education.explanation}</p>
        </div>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 class="text-sm font-bold text-slate-800 mb-2">Medication Note</h4>
            <p class="text-slate-600 font-medium leading-relaxed">${report.education.medication}</p>
          </div>
          <div>
            <h4 class="text-sm font-bold text-slate-800 mb-2">Recommended Next Steps</h4>
            <ul class="space-y-2">`;
    if (report.education.next_steps && report.education.next_steps.length) {
      report.education.next_steps.forEach(step => {
        html += `
          <li class="flex items-start gap-2 text-slate-600 font-medium">
            <svg class="w-5 h-5 text-teal-500 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
            <span>${step}</span>
          </li>`;
      });
    }
    html += `</ul></div></div>`;
    if (report.education.visual_aid_url) {
      html += `
        <div class="mt-6 pt-6 border-t border-slate-100">
          <h4 class="text-sm font-bold text-slate-800 mb-3">Visual Reference</h4>
          ${renderVisualAid(report.education.visual_aid_url, report.diagnosis.primary)}
        </div>`;
    }
    html += `</div></div>`;

    // ── Learn mode card ────────────────────────────────────────────
    if (report.reasoning.length || report.guidelines.length) {
      html += `
      <div class="bg-[#1a2333] rounded-2xl p-6 md:p-8 shadow-inner border-2 border-[#138808]/50 mt-8 relative overflow-hidden">
        <div class="absolute inset-0 opacity-20" style="background-image:radial-gradient(#4b5563 1px,transparent 1px);background-size:16px 16px"></div>
        <div class="relative z-10">
          <div class="flex items-center gap-3 mb-6 border-b border-slate-700 pb-4">
            <div class="bg-indigo-500/20 p-2 rounded-xl text-indigo-400">
              <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"></path></svg>
            </div>
            <div>
              <h3 class="text-xl font-bold text-white">Medical Student Learn Mode</h3>
              <p class="text-xs text-indigo-300 font-semibold uppercase tracking-widest mt-1">AI Reasoning Trace</p>
            </div>
          </div>`;
      if (report.reasoning.length) {
        html += `<div class="mb-6"><h4 class="text-sm font-bold text-indigo-400 mb-3 uppercase tracking-wider">Step-by-step Reasoning</h4><div class="space-y-3">`;
        report.reasoning.forEach((r, idx) => {
          html += `<div class="flex items-start gap-3 bg-slate-800/50 p-3 rounded-lg border border-slate-700/50"><span class="bg-indigo-900 text-indigo-300 text-xs font-bold w-6 h-6 rounded-full flex items-center justify-center shrink-0 mt-0.5">${idx + 1}</span><p class="text-slate-300 font-mono text-sm leading-relaxed">${r}</p></div>`;
        });
        html += `</div></div>`;
      }
      if (report.guidelines.length) {
        html += `<div><h4 class="text-sm font-bold text-[#138808] mb-3 uppercase tracking-wider">Guideline Constraints Context</h4><ul class="space-y-2">`;
        report.guidelines.forEach(g => {
          html += `<li class="flex items-start gap-2 text-slate-400 font-mono text-sm bg-slate-800/30 p-2 rounded border-l-2 border-[#138808]"><span>${g}</span></li>`;
        });
        html += `</ul></div>`;
      }
      // Confidence history chart (learn mode)
      html += renderConfidenceChart(_rawPrimary);
      html += `</div></div>`;
    }

    // ── Commit to DOM ───────────────────────────────────────────────
    reportContent.innerHTML = html;
    reportError.classList.add('hidden');
    if (rateLimitAlert) rateLimitAlert.classList.add('hidden');

    // Stagger card entrance animation
    Array.from(reportContent.children).forEach((card, i) => {
      card.classList.add('report-card-stagger');
      card.style.animationDelay = `${i * 90}ms`;
    });

    // Copy-diagnosis button
    const copyBtn = reportContent.querySelector('.copy-diagnosis-btn');
    if (copyBtn) {
      copyBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(copyBtn.getAttribute('data-copy') || '')
          .then(() => showCopyToast('\u2705 Diagnosis copied!'))
          .catch(() => showCopyToast('\u274c Copy failed — try manually'));
      });
    }

    // Show try-another CTA
    if (tryAnotherSection) tryAnotherSection.classList.remove('hidden');

    // Advanced debug section
    let adv = '';
    adv += `<div class="mb-4 bg-slate-800 p-4 rounded-xl border border-slate-700">`;
    adv += `  <div class='mb-1 text-xs text-slate-500 uppercase font-bold tracking-widest'>Computed Bias Risk Score</div>`;
    adv += `  <div class='text-2xl font-black ${parseFloat(report.equity_check.bias_score) > 0.5 ? 'text-red-500' : 'text-green-500'}'>${report.equity_check.bias_score}</div>`;
    adv += `</div>`;
    if (report.equity_check.potential_biases && report.equity_check.potential_biases.length) {
      adv += `<div class='mb-6'><div class='mb-2 text-sm font-bold text-orange-400 uppercase tracking-wider'>Identified Potential Biases</div><ul class='space-y-2'>`;
      report.equity_check.potential_biases.forEach(b => { adv += `<li class="bg-slate-800/50 p-3 rounded border-l-4 border-orange-500 text-slate-300 font-mono text-sm">${b}</li>`; });
      adv += `</ul></div>`;
    }
    if (report.equity_check.cultural_adaptations && report.equity_check.cultural_adaptations.length) {
      adv += `<div class='mb-6'><div class='mb-2 text-sm font-bold text-blue-400 uppercase tracking-wider'>Suggested Cultural Adaptations</div><ul class='space-y-2'>`;
      report.equity_check.cultural_adaptations.forEach(a => { adv += `<li class="bg-slate-800/50 p-3 rounded border-l-4 border-blue-500 text-slate-300 font-mono text-sm">${a}</li>`; });
      adv += `</ul></div>`;
    }
    adv += `<div class='mt-8 pt-6 border-t border-slate-700'><div class='mb-2 text-xs font-bold text-slate-500 uppercase tracking-widest'>Raw Debug Telemetry</div><pre class='bg-black/50 border border-slate-800 p-4 rounded-xl text-xs text-green-400 overflow-x-auto font-mono'>${JSON.stringify(report.debug_info, null, 2)}</pre></div>`;
    advancedContent.innerHTML = adv;
    advancedSection.classList.add('hidden');
    if (toggleAdvancedBtn) toggleAdvancedBtn.textContent = 'View Analytics & Trace';
  }

  // ── Share button handler ──────────────────────────────────────────────────
  if (shareBtn) {
    shareBtn.addEventListener('click', function () {
      const diag = currentReport?.diagnosis?.primary || 'N/A';
      const conf = currentReport?.diagnosis?.confidence != null
        ? `${(parseFloat(currentReport.diagnosis.confidence) * 100).toFixed(0)}%`
        : '';
      const text = `SwasthyaSetu AI Diagnosis: ${diag}${conf ? ` (confidence: ${conf})` : ''}\n\nFor informational purposes only. Always consult a qualified physician in India.`;
      if (navigator.share) {
        navigator.share({ title: 'SwasthyaSetu Medical Report', text }).catch(() => { });
      } else {
        navigator.clipboard.writeText(text)
          .then(() => showCopyToast('\u2705 Report summary copied!'))
          .catch(() => showCopyToast('\u274c Copy failed'));
      }
    });
  }

  // ── Try-another button handler ────────────────────────────────────────────
  if (tryAnotherBtn) {
    tryAnotherBtn.addEventListener('click', function () {
      reportSection.classList.add('hidden');
      if (emergencyBanner) emergencyBanner.classList.add('hidden');
      if (tryAnotherSection) tryAnotherSection.classList.add('hidden');
      symptomsInput.value = '';
      locationInput.value = '';
      updateCharCounter();
      if (symptomsError) symptomsError.classList.add('hidden');
      if (locationError) locationError.classList.add('hidden');
      window.scrollTo({ top: 0, behavior: 'smooth' });
      setTimeout(() => symptomsInput.focus(), 600);
    });
  }

  /**
   * PDF Styles - Print-optimized CSS for html2pdf.js
   * Includes medical-grade styling with Indian tricolor theme
   */
  const pdfStyles = `
    <style>
      /* Reset and base styles */
      * { box-sizing: border-box; }
      body { 
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #1e293b;
        line-height: 1.6;
        margin: 0;
        padding: 0;
      }
      
      /* PDF Container */
      .pdf-container {
        max-width: 210mm;
        margin: 0 auto;
        padding: 20mm;
        background: #ffffff;
      }
      
      /* Header */
      .pdf-header {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        color: white;
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 25px;
        position: relative;
        border-bottom: 4px solid #FF9933;
      }
      
      .pdf-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #FF9933 33.33%, #ffffff 33.33%, #ffffff 66.66%, #138808 66.66%);
        border-radius: 12px 12px 0 0;
      }
      
      .pdf-header-content {
        display: flex;
        align-items: center;
        gap: 15px;
      }
      
      .pdf-logo {
        width: 50px;
        height: 50px;
        background: linear-gradient(135deg, #138808, #0A5C05);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 2px solid white;
      }
      
      .pdf-logo svg {
        width: 28px;
        height: 28px;
        fill: white;
      }
      
      .pdf-title-block h1 {
        margin: 0;
        font-size: 22px;
        font-weight: 800;
        letter-spacing: -0.5px;
      }
      
      .pdf-title-block p {
        margin: 4px 0 0 0;
        font-size: 12px;
        opacity: 0.8;
        font-weight: 500;
      }
      
      .pdf-meta {
        margin-left: auto;
        text-align: right;
        font-size: 11px;
        opacity: 0.9;
      }
      
      .pdf-meta .pdf-date {
        font-weight: 600;
        font-size: 12px;
      }
      
      /* Report Title */
      .pdf-report-title {
        text-align: center;
        margin-bottom: 25px;
        padding-bottom: 15px;
        border-bottom: 2px solid #e2e8f0;
      }
      
      .pdf-report-title h2 {
        margin: 0;
        font-size: 24px;
        font-weight: 800;
        color: #000080;
      }
      
      .pdf-report-title p {
        margin: 5px 0 0 0;
        font-size: 12px;
        color: #64748b;
      }
      
      /* Card Styles */
      .pdf-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        page-break-inside: avoid;
      }
      
      .pdf-card-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 15px;
        padding-bottom: 12px;
        border-bottom: 1px solid #f1f5f9;
      }
      
      .pdf-card-icon {
        width: 40px;
        height: 40px;
        background: #dbeafe;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #000080;
      }
      
      .pdf-card-icon svg {
        width: 22px;
        height: 22px;
      }
      
      .pdf-card-icon.green {
        background: #dcfce7;
        color: #138808;
      }
      
      .pdf-card-icon.orange {
        background: #ffedd5;
        color: #FF9933;
      }
      
      .pdf-card-icon.red {
        background: #fee2e2;
        color: #dc2626;
      }
      
      .pdf-card-title {
        margin: 0;
        font-size: 16px;
        font-weight: 700;
        color: #1e293b;
      }
      
      /* Primary Diagnosis */
      .pdf-diagnosis-primary {
        display: flex;
        align-items: center;
        gap: 12px;
        flex-wrap: wrap;
        margin-bottom: 12px;
      }
      
      .pdf-diagnosis-name {
        font-size: 26px;
        font-weight: 800;
        color: #000080;
        margin: 0;
      }
      
      .pdf-confidence-badge {
        background: #dcfce7;
        color: #138808;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        border: 1px solid #86efac;
      }
      
      .pdf-alternatives {
        margin-top: 15px;
        padding-top: 15px;
        border-top: 1px solid #f1f5f9;
      }
      
      .pdf-alternatives-label {
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #94a3b8;
        font-weight: 700;
        margin-bottom: 8px;
      }
      
      .pdf-alt-tag {
        display: inline-block;
        background: #f8fafc;
        color: #475569;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
        border: 1px solid #e2e8f0;
        margin: 2px;
      }
      
      .pdf-validation {
        margin-top: 15px;
        padding-top: 15px;
        border-top: 1px solid #f1f5f9;
        display: flex;
        justify-content: space-between;
        align-items: center;
      }
      
      .pdf-validation-label {
        font-size: 12px;
        color: #64748b;
        font-weight: 600;
      }
      
      .pdf-validation-status {
        font-size: 12px;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 6px;
      }
      
      .pdf-validation-status.confirmed {
        background: #dcfce7;
        color: #138808;
      }
      
      .pdf-validation-status.pending {
        background: #ffedd5;
        color: #c2410c;
      }
      
      /* Triage Section */
      .pdf-triage-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
      }
      
      .pdf-triage-item-label {
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #64748b;
        font-weight: 700;
        margin-bottom: 4px;
      }
      
      .pdf-triage-value {
        font-size: 18px;
        font-weight: 800;
        margin: 0;
      }
      
      .pdf-triage-value.emergency { color: #dc2626; }
      .pdf-triage-value.urgent { color: #FF9933; }
      .pdf-triage-value.normal { color: #138808; }
      
      .pdf-triage-next-step {
        font-size: 13px;
        font-weight: 700;
        color: #1e293b;
        margin: 0;
      }
      
      .pdf-triage-explanation {
        margin-top: 15px;
        padding-top: 15px;
        border-top: 1px solid #f1f5f9;
        font-size: 13px;
        color: #475569;
        line-height: 1.6;
      }
      
      /* Providers Grid */
      .pdf-providers-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
      }
      
      .pdf-provider-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
      }
      
      .pdf-provider-name {
        font-size: 12px;
        font-weight: 700;
        color: #1e293b;
        margin: 0 0 6px 0;
        line-height: 1.4;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
      }
      
      .pdf-provider-link {
        font-size: 10px;
        color: #000080;
        font-weight: 600;
      }
      
      .pdf-no-results {
        text-align: center;
        padding: 20px;
        background: #f8fafc;
        border-radius: 8px;
        color: #64748b;
        font-size: 13px;
      }
      
      /* Education Section */
      .pdf-education-block {
        margin-bottom: 15px;
      }
      
      .pdf-education-block:last-child {
        margin-bottom: 0;
      }
      
      .pdf-edu-label {
        font-size: 12px;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 6px;
      }
      
      .pdf-edu-content {
        font-size: 13px;
        color: #475569;
        line-height: 1.6;
        margin: 0;
      }
      
      .pdf-edu-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
      }
      
      .pdf-next-steps-list {
        list-style: none;
        padding: 0;
        margin: 0;
      }
      
      .pdf-next-steps-list li {
        display: flex;
        align-items: flex-start;
        gap: 8px;
        margin-bottom: 8px;
        font-size: 13px;
        color: #475569;
      }
      
      .pdf-check-icon {
        width: 18px;
        height: 18px;
        color: #138808;
        flex-shrink: 0;
        margin-top: 2px;
      }
      
      .pdf-edu-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        margin-top: 15px;
      }
      
      /* Visual Aid in PDF */
      .pdf-visual-aid {
        margin-top: 15px;
        padding: 15px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        text-align: center;
      }
      
      .pdf-visual-aid img {
        max-width: 100%;
        height: auto;
        border-radius: 6px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      }
      
      .pdf-visual-caption {
        margin-top: 10px;
        font-size: 10px;
        color: #64748b;
        font-style: italic;
      }
      
      /* Footer */
      .pdf-footer {
        margin-top: 30px;
        padding: 20px;
        background: linear-gradient(135deg, #fef3c7 0%, #fffbeb 100%);
        border: 1px solid #fde68a;
        border-radius: 12px;
        border-left: 4px solid #FF9933;
      }
      
      .pdf-footer-title {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 13px;
        font-weight: 800;
        color: #92400e;
        margin-bottom: 8px;
      }
      
      .pdf-footer-text {
        font-size: 11px;
        color: #78350f;
        line-height: 1.5;
        margin: 0;
      }
      
      .pdf-footer strong {
        color: #451a03;
      }
      
      /* Page Break */
      .pdf-page-break {
        page-break-before: always;
      }
      
      /* Utility */
      .pdf-hidden-print {
        display: none !important;
      }
      
      /* Print optimization */
      @media print {
        .pdf-container {
          padding: 15mm;
        }
        
        .pdf-card {
          break-inside: avoid;
        }
        
        .pdf-header {
          break-inside: avoid;
        }
      }
    </style>
  `;

  /**
   * Generate and download PDF using html2pdf.js
   * Creates a professional medical-grade PDF report
   */
  downloadBtn.addEventListener('click', function () {
    // Validate report content exists
    if (!reportContent.innerHTML.trim()) {
      console.error('No report content to generate PDF');
      return;
    }

    // Get current date formatted for India timezone
    const reportDate = new Date().toLocaleString('en-IN', {
      timeZone: 'Asia/Kolkata',
      dateStyle: 'long',
      timeStyle: 'short'
    }) + ' IST';

    // Build PDF HTML content with proper styling
    const pdfContent = `
      <!DOCTYPE html>
      <html lang="en">
      <head>
        <meta charset="UTF-8">
        <title>SwasthyaSetu Medical Diagnosis Report</title>
        ${pdfStyles}
      </head>
      <body>
        <div class="pdf-container">
          <!-- Header with Logo -->
          <div class="pdf-header">
            <div class="pdf-header-content">
              <div class="pdf-logo">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="white">
                  <path fill-rule="evenodd" d="M12 2.25c-5.385 0-9.75 4.365-9.75 9.75s4.365 9.75 9.75 9.75 9.75-4.365 9.75-9.75S17.385 2.25 12 2.25zM12.75 8a.75.75 0 00-1.5 0v3.25H8a.75.75 0 000 1.5h3.25V16a.75.75 0 001.5 0v-3.25H16a.75.75 0 000-1.5h-3.25V8z" clip-rule="evenodd" />
                </svg>
              </div>
              <div class="pdf-title-block">
                <h1>SwasthyaSetu</h1>
                <p>Bharat's AI Medical Assistant</p>
              </div>
              <div class="pdf-meta">
                <div class="pdf-date">${reportDate}</div>
                <div>Report ID: ${Math.random().toString(36).substring(2, 10).toUpperCase()}</div>
              </div>
            </div>
          </div>

          <!-- Report Title -->
          <div class="pdf-report-title">
            <h2>Medical Diagnosis Report</h2>
            <p>AI-Generated Health Assessment • Confidential</p>
          </div>

          <!-- Report Content -->
          <div id="pdf-content-area">
            ${reportContent.innerHTML}
          </div>

          <!-- Footer Disclaimer -->
          <div class="pdf-footer">
            <div class="pdf-footer-title">
              <svg width="16" height="16" fill="currentColor" viewBox="0 0 20 20">
                <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/>
              </svg>
              Medical Disclaimer
            </div>
            <p class="pdf-footer-text">
              <strong>Important:</strong> SwasthyaSetu AI is an experimental artificial intelligence system for informational purposes only. 
              This report is <strong>NOT</strong> a substitute for professional medical diagnosis, advice, or treatment. 
              Always consult a qualified physician or healthcare provider in India for proper evaluation of your health conditions. 
              In case of emergency, immediately contact your nearest hospital or dial emergency services (102/108/112).
            </p>
          </div>
        </div>
      </body>
      </html>
    `;

    // Create temporary container for PDF generation
    const tempContainer = document.createElement('div');
    tempContainer.innerHTML = pdfContent;
    tempContainer.style.position = 'absolute';
    tempContainer.style.left = '-9999px';
    tempContainer.style.top = '0';
    document.body.appendChild(tempContainer);

    // Clean up Tailwind classes and convert to PDF-friendly structure
    const contentArea = tempContainer.querySelector('#pdf-content-area');

    // Transform diagnosis card
    const diagnosisCards = contentArea.querySelectorAll('.bg-white.rounded-2xl');
    diagnosisCards.forEach(card => {
      // Convert Tailwind classes to PDF classes
      card.className = 'pdf-card';

      // Transform card header
      const header = card.querySelector('.flex.items-center.gap-3.mb-4, .flex.items-center.gap-3.mb-5');
      if (header) {
        header.className = 'pdf-card-header';
        const icon = header.querySelector('div[class*="rounded-xl"], div[class*="rounded-2xl"]');
        if (icon) {
          // Determine icon type based on content
          const titleText = header.textContent.toLowerCase();
          if (titleText.includes('diagnosis')) {
            icon.className = 'pdf-card-icon';
          } else if (titleText.includes('triage')) {
            icon.className = 'pdf-card-icon orange';
          } else if (titleText.includes('provider')) {
            icon.className = 'pdf-card-icon';
          } else if (titleText.includes('understanding')) {
            icon.className = 'pdf-card-icon green';
          } else {
            icon.className = 'pdf-card-icon';
          }
        }
        const title = header.querySelector('h3');
        if (title) title.className = 'pdf-card-title';
      }

      // Transform diagnosis name and confidence
      const diagnosisRow = card.querySelector('.flex.flex-col.sm\\:flex-row');
      if (diagnosisRow) {
        diagnosisRow.className = 'pdf-diagnosis-primary';
        const nameEl = diagnosisRow.querySelector('span.text-3xl, span.text-4xl');
        if (nameEl) nameEl.className = 'pdf-diagnosis-name';
        const confidenceEl = diagnosisRow.querySelector('span.inline-flex');
        if (confidenceEl) {
          confidenceEl.className = 'pdf-confidence-badge';
          // Remove SVG icons for cleaner PDF
          const svg = confidenceEl.querySelector('svg');
          if (svg) svg.remove();
        }
      }

      // Transform alternatives
      const alternatives = card.querySelector('.flex.flex-wrap.gap-2');
      if (alternatives) {
        alternatives.className = 'pdf-alternatives';
        const parent = alternatives.parentElement;
        if (parent && parent.querySelector('span.text-xs')) {
          parent.querySelector('span.text-xs').className = 'pdf-alternatives-label';
        }
        alternatives.querySelectorAll('span').forEach(tag => {
          tag.className = 'pdf-alt-tag';
        });
      }

      // Transform validation status
      const validation = card.querySelector('.flex.items-center.justify-between');
      if (validation && validation.textContent.includes('Validation')) {
        validation.className = 'pdf-validation';
        const statusLabel = validation.querySelector('span.text-sm.font-semibold');
        if (statusLabel) statusLabel.className = 'pdf-validation-label';
        const statusValue = validation.querySelector('span.font-bold:last-child');
        if (statusValue) {
          const isConfirmed = statusValue.textContent.includes('Confirmed');
          statusValue.className = 'pdf-validation-status ' + (isConfirmed ? 'confirmed' : 'pending');
        }
      }

      // Transform triage grid
      const triageGrid = card.querySelector('.grid.grid-cols-1.md\\:grid-cols-2');
      if (triageGrid && triageGrid.querySelector('.text-xs')) {
        triageGrid.className = 'pdf-triage-grid';
        const items = triageGrid.querySelectorAll('div');
        items.forEach(item => {
          const label = item.querySelector('.text-xs');
          const value = item.querySelector('.text-2xl');
          const nextStep = item.querySelector('.text-lg');
          if (label) label.className = 'pdf-triage-item-label';
          if (value) {
            const levelText = value.textContent.toLowerCase();
            let levelClass = 'normal';
            if (levelText.includes('emergency')) levelClass = 'emergency';
            else if (levelText.includes('urgent')) levelClass = 'urgent';
            value.className = 'pdf-triage-value ' + levelClass;
          }
          if (nextStep) nextStep.className = 'pdf-triage-next-step';
        });
      }

      // Transform triage explanation
      const triageExp = card.querySelector('.mt-4.pt-4.border-t p');
      if (triageExp && card.textContent.includes('Triage')) {
        const container = triageExp.closest('.mt-4');
        if (container) {
          container.className = 'pdf-triage-explanation';
          triageExp.className = '';
        }
      }

      // Transform providers grid
      const providersGrid = card.querySelector('.grid.grid-cols-1.sm\\:grid-cols-2.lg\\:grid-cols-3');
      if (providersGrid) {
        providersGrid.className = 'pdf-providers-grid';
        providersGrid.querySelectorAll('a').forEach(link => {
          link.className = 'pdf-provider-card';
          link.removeAttribute('target');
          link.removeAttribute('rel');
          const title = link.querySelector('h4');
          if (title) title.className = 'pdf-provider-name';
          const viewLink = link.querySelector('.text-xs');
          if (viewLink) {
            viewLink.className = 'pdf-provider-link';
            viewLink.innerHTML = 'View Details →';
          }
        });
      }

      // Transform no results message
      const noResults = card.querySelector('.bg-slate-50.border.border-slate-200.p-4');
      if (noResults) {
        noResults.className = 'pdf-no-results';
      }

      // Transform education sections
      const eduBlocks = card.querySelectorAll('.space-y-6 > div');
      eduBlocks.forEach(block => {
        block.className = 'pdf-education-block';
        const label = block.querySelector('h4');
        const content = block.querySelector('p');
        const box = block.querySelector('.bg-slate-50');
        if (label) label.className = 'pdf-edu-label';
        if (content) content.className = 'pdf-edu-content';
        if (box) box.className = 'pdf-edu-box';
      });

      // Transform next steps list
      const nextStepsList = card.querySelector('ul.space-y-2');
      if (nextStepsList) {
        nextStepsList.className = 'pdf-next-steps-list';
        nextStepsList.querySelectorAll('li').forEach(li => {
          li.innerHTML = li.innerHTML.replace(/<svg[^>]*class="w-5[^"]*"[^>]*>/, '<svg class="pdf-check-icon" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path></svg>');
        });
      }

      // Transform education grid
      const eduGrid = card.querySelector('.grid.grid-cols-1.md\\:grid-cols-2.gap-6');
      if (eduGrid) {
        eduGrid.className = 'pdf-edu-grid';
      }

      // Transform visual aid for PDF
      const visualAid = card.querySelector('.visual-aid-container');
      if (visualAid) {
        const img = visualAid.querySelector('img');
        if (img && img.src) {
          visualAid.className = 'pdf-visual-aid';
          visualAid.innerHTML = `
            <img src="${img.src}" alt="${img.alt || 'Medical illustration'}" style="max-width: 300px;" />
            <div class="pdf-visual-caption">AI-Generated Educational Illustration</div>
          `;
        }
      }
    });

    // Remove dark mode / learn mode sections from PDF (too complex)
    const darkSections = contentArea.querySelectorAll('.bg-\\[\\#1a2333\\], .bg-\\[\\#0A0F1C\\]');
    darkSections.forEach(section => section.remove());

    // Configuration for html2pdf
    const pdfOptions = {
      margin: 0,
      filename: `swasthyasetu_medical_report_${new Date().toISOString().split('T')[0]}.pdf`,
      image: { type: 'jpeg', quality: 0.98 },
      html2canvas: {
        scale: 2,
        useCORS: true,
        logging: false,
        letterRendering: true,
        allowTaint: true,
        backgroundColor: '#ffffff'
      },
      jsPDF: {
        unit: 'mm',
        format: 'a4',
        orientation: 'portrait',
        compress: true
      },
      pagebreak: {
        mode: ['avoid-all', 'css', 'legacy'],
        before: '.pdf-page-break',
        avoid: '.pdf-card'
      }
    };

    // Generate PDF
    html2pdf()
      .set(pdfOptions)
      .from(tempContainer)
      .save()
      .then(() => {
        // Cleanup temporary container
        document.body.removeChild(tempContainer);
        console.log('PDF generated successfully');
      })
      .catch(error => {
        document.body.removeChild(tempContainer);
        console.error('PDF generation failed:', error);
        alert('Failed to generate PDF. Please try using the Print button instead.');
      });
  });

  /**
   * Print handler - Opens report in print window
   */
  printBtn.addEventListener('click', function () {
    const printWindow = window.open('', '_blank');
    const printDate = new Date().toLocaleString('en-IN', {
      timeZone: 'Asia/Kolkata',
      dateStyle: 'long',
      timeStyle: 'short'
    }) + ' IST';

    printWindow.document.write(`
      <!DOCTYPE html>
      <html>
      <head>
        <title>SwasthyaSetu Medical Diagnosis Report</title>
        ${pdfStyles}
      </head>
      <body>
        <div class="pdf-container">
          <div class="pdf-header">
            <div class="pdf-header-content">
              <div class="pdf-logo">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="white">
                  <path fill-rule="evenodd" d="M12 2.25c-5.385 0-9.75 4.365-9.75 9.75s4.365 9.75 9.75 9.75 9.75-4.365 9.75-9.75S17.385 2.25 12 2.25zM12.75 8a.75.75 0 00-1.5 0v3.25H8a.75.75 0 000 1.5h3.25V16a.75.75 0 001.5 0v-3.25H16a.75.75 0 000-1.5h-3.25V8z" clip-rule="evenodd" />
                </svg>
              </div>
              <div class="pdf-title-block">
                <h1>SwasthyaSetu</h1>
                <p>Bharat's AI Medical Assistant</p>
              </div>
              <div class="pdf-meta">
                <div class="pdf-date">${printDate}</div>
              </div>
            </div>
          </div>
          <div class="pdf-report-title">
            <h2>Medical Diagnosis Report</h2>
          </div>
          ${reportContent.innerHTML}
          <div class="pdf-footer">
            <div class="pdf-footer-title">Medical Disclaimer</div>
            <p class="pdf-footer-text">
              <strong>Important:</strong> This is an AI-generated report for informational purposes only and NOT a substitute for professional medical diagnosis. Always consult a qualified physician.
            </p>
          </div>
        </div>
      </body>
      </html>
    `);
    printWindow.document.close();
    printWindow.focus();
    setTimeout(() => printWindow.print(), 500);
  });
});
