document.addEventListener('DOMContentLoaded', function() {
  const form = document.getElementById('diagnosis-form');
  const symptomsInput = document.getElementById('symptoms');
  const locationInput = document.getElementById('location');
  const learnModeInput = document.getElementById('learn-mode');
  const loading = document.getElementById('loading');
  const reportSection = document.getElementById('report-section');
  const reportContent = document.getElementById('report-content');
  const reportError = document.getElementById('report-error');
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

  if (toggleAdvancedBtn && advancedSection) {
    toggleAdvancedBtn.addEventListener('click', function() {
      if (advancedSection.classList.contains('hidden')) {
        advancedSection.classList.remove('hidden');
        toggleAdvancedBtn.textContent = 'Hide Advanced Details';
      } else {
        advancedSection.classList.add('hidden');
        toggleAdvancedBtn.textContent = 'Show Advanced Details';
      }
    });
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

  function setProgress(stepIdx) {
    progressIndicator.classList.remove('hidden');
    const percent = Math.round((stepIdx / (steps.length - 1)) * 100);
    progressBar.style.width = percent + '%';
    progressStep.textContent = steps[stepIdx] || '';
  }

  function resetProgress() {
    progressIndicator.classList.add('hidden');
    progressBar.style.width = '0%';
    progressStep.textContent = '';
  }

  form.addEventListener('submit', async function(e) {
    e.preventDefault();
    const symptoms = symptomsInput.value.trim();
    const location = locationInput.value.trim();
    const learn_mode = learnModeInput && learnModeInput.checked;
    let valid = true;
    if (!symptoms) {
      symptomsInput.focus();
      symptomsInput.setAttribute('aria-invalid', 'true');
      valid = false;
    } else {
      symptomsInput.setAttribute('aria-invalid', 'false');
    }
    if (!location) {
      locationInput.focus();
      locationInput.setAttribute('aria-invalid', 'true');
      valid = false;
    } else {
      locationInput.setAttribute('aria-invalid', 'false');
    }
    if (!valid) return;
    loading.classList.remove('hidden');
    resetProgress();
    setProgress(1);
    reportSection.classList.add('hidden');
    reportError.classList.add('hidden');
    reportContent.innerHTML = '';
    try {
      setProgress(2);
      const response = await fetch('/diagnose', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symptoms, location, learn_mode })
      });
      setProgress(4);
      const data = await response.json();
      setProgress(7);
      if (data.status === 'success') {
        renderReport(data.report);
        setProgress(8);
        reportSection.classList.remove('hidden');
      } else {
        showError(data.error || 'Unknown error.');
      }
    } catch (err) {
      showError(err.message);
    } finally {
      loading.classList.add('hidden');
      setTimeout(resetProgress, 1200);
    }
  });

  function showError(msg) {
    reportError.textContent = 'Error: ' + msg;
    reportError.classList.remove('hidden');
    reportSection.classList.remove('hidden');
  }

  exampleBtn.addEventListener('click', function() {
    symptomsInput.value = 'Patient presents with high fever, chills, headache, and fatigue. Recently returned from a trip to a tropical region known for mosquito-borne illnesses.';
    symptomsInput.focus();
  });

  clearBtn.addEventListener('click', function() {
    symptomsInput.value = '';
    locationInput.value = '';
    symptomsInput.focus();
    reportSection.classList.add('hidden');
    reportContent.innerHTML = '';
    reportError.classList.add('hidden');
  });

  exampleChips.forEach(chip => {
    chip.addEventListener('click', function() {
      symptomsInput.value = chip.textContent;
      symptomsInput.focus();
    });
    chip.addEventListener('keydown', function(e) {
      if (e.key === 'Enter' || e.key === ' ') {
        symptomsInput.value = chip.textContent;
        symptomsInput.focus();
      }
    });
  });

  function renderReport(report) {
    if (!report || !report.diagnosis) {
      showError('No report data.');
      return;
    }
    let html = '';
    // Diagnosis
    html += `<div class='mb-4'><span class='font-semibold'>Primary Diagnosis:</span> <span class='text-blue-700 text-lg font-bold'>${report.diagnosis.primary}</span> <span class='ml-2 text-sm text-gray-500'>(Confidence: ${(report.diagnosis.confidence*100).toFixed(1)}%)</span></div>`;
    if (report.diagnosis.alternatives && report.diagnosis.alternatives.length) {
      html += `<div class='mb-2'><span class='font-semibold'>Alternatives:</span> <span class='text-gray-700'>${report.diagnosis.alternatives.join(', ')}</span></div>`;
    }
    html += `<div class='mb-2'><span class='font-semibold'>Validation Status:</span> <span class='text-green-700'>${report.diagnosis.validation_status}</span></div>`;
    // Triage
    if (report.triage) {
      html += `<hr class='my-4'>`;
      html += `<div class='mb-2'><span class='font-semibold'>Urgency Level:</span> <span class='text-red-700 font-bold'>${report.triage.level}</span></div>`;
      html += `<div class='mb-2'><span class='font-semibold'>Next Step:</span> <span class='text-blue-800'>${report.triage.next_step}</span></div>`;
      html += `<div class='mb-2'><span class='font-semibold'>Triage Explanation:</span> <span class='text-gray-800'>${report.triage.explanation}</span></div>`;
    }
    // Routing
    if (report.routing) {
      html += `<hr class='my-4'>`;
      html += `<div class='mb-2'><span class='font-semibold'>Local Doctor/Clinic Suggestions:</span></div>`;
      if (report.routing.results && report.routing.results.length) {
        html += `<ul class='list-disc ml-6'>`;
        report.routing.results.forEach(r => {
          html += `<li><a href='${r.url}' target='_blank' class='text-blue-600 underline'>${r.title}</a></li>`;
        });
        html += `</ul>`;
      } else {
        html += `<div class='text-gray-500'>No local results found. Try refining your location.</div>`;
      }
    }
    // Education
    html += `<hr class='my-4'>`;
    html += `<div class='mb-2'><span class='font-semibold'>Explanation:</span> <span class='text-gray-800'>${report.education.explanation}</span></div>`;
    html += `<div class='mb-2'><span class='font-semibold'>Medication Info:</span> <span class='text-gray-800'>${report.education.medication}</span></div>`;
    if (report.education.next_steps && report.education.next_steps.length) {
      html += `<div class='mb-2'><span class='font-semibold'>Next Steps:</span><ul class='list-disc ml-6'>`;
      report.education.next_steps.forEach(step => {
        html += `<li>${step}</li>`;
      });
      html += `</ul></div>`;
    }
    html += `<div class='mb-2'><span class='font-semibold'>Visual Aid:</span> <span class='italic text-gray-600'>${report.education.visual}</span></div>`;
    html += `<div class='mt-4 text-xs text-gray-400'>Status: ${report.workflow_status}</div>`;
    // Learn Mode: Show reasoning/guidelines if present
    if (report.reasoning || report.guidelines) {
      html += `<hr class='my-4'>`;
      if (report.reasoning) {
        html += `<div class='mb-2'><span class='font-semibold text-blue-700'>Step-by-step Reasoning:</span><ul class='list-decimal ml-6 text-gray-800'>`;
        report.reasoning.forEach(r => { html += `<li>${r}</li>`; });
        html += `</ul></div>`;
      }
      if (report.guidelines) {
        html += `<div class='mb-2'><span class='font-semibold text-green-700'>Guideline References:</span><ul class='list-disc ml-6 text-gray-700'>`;
        report.guidelines.forEach(g => { html += `<li>${g}</li>`; });
        html += `</ul></div>`;
      }
    }
    reportContent.innerHTML = html;
    reportError.classList.add('hidden');
    // Advanced section (hidden by default)
    let adv = '';
    adv += `<div class='mb-2'><span class='font-semibold'>Bias Risk Score:</span> <span class='text-yellow-700'>${report.equity_check.bias_score}</span></div>`;
    if (report.equity_check.potential_biases && report.equity_check.potential_biases.length) {
      adv += `<div class='mb-2'><span class='font-semibold'>Potential Biases:</span><ul class='list-disc ml-6'>`;
      report.equity_check.potential_biases.forEach(bias => {
        adv += `<li>${bias}</li>`;
      });
      adv += `</ul></div>`;
    }
    if (report.equity_check.cultural_adaptations && report.equity_check.cultural_adaptations.length) {
      adv += `<div class='mb-2'><span class='font-semibold'>Cultural Adaptations:</span><ul class='list-disc ml-6'>`;
      report.equity_check.cultural_adaptations.forEach(adapt => {
        adv += `<li>${adapt}</li>`;
      });
      adv += `</ul></div>`;
    }
    adv += `<div class='mb-2'><span class='font-semibold'>Debug Info:</span> <pre class='bg-gray-100 rounded p-2 text-xs'>${JSON.stringify(report.debug_info, null, 2)}</pre></div>`;
    advancedContent.innerHTML = adv;
    advancedSection.classList.add('hidden');
    if (toggleAdvancedBtn) toggleAdvancedBtn.textContent = 'Show Advanced Details';
  }

  downloadBtn.addEventListener('click', function() {
    const content = reportContent.innerHTML;
    const blob = new Blob([
      '<html><head><meta charset="UTF-8"><title>Diagnosis Report</title><style>body{font-family:sans-serif;padding:2em;}h2{color:#2563eb;}</style></head><body>' +
      '<h2>Diagnosis Report</h2>' + content + '</body></html>'
    ], {type: 'text/html'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'swasthyasetu_diagnosis_report.html';
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { document.body.removeChild(a); URL.revokeObjectURL(url); }, 100);
  });

  printBtn.addEventListener('click', function() {
    const printWindow = window.open('', '_blank');
    printWindow.document.write('<html><head><title>Diagnosis Report</title><style>body{font-family:sans-serif;padding:2em;}h2{color:#2563eb;}</style></head><body>' +
      '<h2>Diagnosis Report</h2>' + reportContent.innerHTML + '</body></html>');
    printWindow.document.close();
    printWindow.focus();
    printWindow.print();
  });
});
