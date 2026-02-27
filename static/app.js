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
    
    // Diagnosis Card
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
           <span class="inline-flex items-center gap-1 bg-green-100 text-[#138808] text-sm font-bold px-3 py-1 rounded-full border border-green-200">
             <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path></svg>
             Confidence: ${(report.diagnosis.confidence*100).toFixed(1)}%
           </span>
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
           <span class="text-sm font-bold ${report.diagnosis.validation_status.includes('Confirmed') ? 'text-[#138808]' : 'text-[#FF9933]'}">${report.diagnosis.validation_status}</span>
         </div>
       </div>
    </div>
    `;

    // Triage Card
    if (report.triage) {
      const isEmergency = report.triage.level.toLowerCase().includes('emergency');
      const isUrgent = report.triage.level.toLowerCase().includes('urgent');
      
      let levelColor = 'text-[#138808]';
      let bgColor = 'bg-green-50/50';
      let iconColor = 'text-[#138808]';
      let borderColor = 'border-green-100';
      
      if (isEmergency) {
        levelColor = 'text-red-600';
        bgColor = 'bg-red-50/50';
        iconColor = 'text-red-500';
        borderColor = 'border-red-200';
      } else if (isUrgent) {
        levelColor = 'text-[#FF9933]';
        bgColor = 'bg-orange-50/50';
        iconColor = 'text-[#FF9933]';
        borderColor = 'border-orange-200';
      }

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
            <div class="text-2xl font-extrabold capitalize ${levelColor}">${report.triage.level}</div>
          </div>
          <div>
            <div class="text-xs font-bold text-slate-500 uppercase tracking-wider mb-1">Recommended Action</div>
            <div class="text-lg font-bold text-slate-700">${report.triage.next_step}</div>
          </div>
        </div>
        
        <div class="mt-4 pt-4 border-t border-black/5">
          <p class="text-slate-600 font-medium leading-relaxed">${report.triage.explanation}</p>
        </div>
      </div>
      `;
    }

    // Routing Card (Doctors/Hospitals in India)
    if (report.routing) {
      html += `
      <div class="bg-white rounded-2xl p-6 md:p-8 shadow-sm border border-slate-100 mb-6">
        <div class="flex items-center gap-3 mb-5">
          <div class="bg-indigo-50 p-2 rounded-xl text-indigo-600">
             <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4"></path></svg>
          </div>
          <h3 class="text-xl font-bold text-slate-800">Local Healthcare Providers</h3>
        </div>
      `;
      
      if (report.routing.results && report.routing.results.length) {
        html += `<div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">`;
        report.routing.results.forEach(r => {
          html += `
            <a href="${r.url}" target="_blank" class="block bg-slate-50 hover:bg-indigo-50 border border-slate-200 hover:border-indigo-200 p-4 rounded-xl transition-all group shadow-sm hover:shadow">
              <h4 class="font-bold text-slate-800 group-hover:text-indigo-700 line-clamp-2 mb-2 text-sm">${r.title}</h4>
              <div class="flex items-center text-xs font-semibold text-indigo-600 group-hover:text-indigo-800">
                View Search Result <svg class="w-3 h-3 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"></path></svg>
              </div>
            </a>
          `;
        });
        html += `</div>`;
      } else {
        html += `
          <div class="bg-slate-50 border border-slate-200 p-4 rounded-xl text-center">
             <svg class="w-8 h-8 text-slate-400 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7"></path></svg>
             <p class="text-slate-500 font-medium text-sm">No specific local results found. Try refining your location or city name above.</p>
          </div>
        `;
      }
      html += `</div>`;
    }

    // Education & Next Steps Card
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
             <ul class="space-y-2">
      `;
      
      if (report.education.next_steps && report.education.next_steps.length) {
        report.education.next_steps.forEach(step => {
          html += `
            <li class="flex items-start gap-2 text-slate-600 font-medium">
              <svg class="w-5 h-5 text-teal-500 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
              <span>${step}</span>
            </li>
          `;
        });
      }
      
      html += `
             </ul>
          </div>
        </div>
      </div>
    </div>
    `;

    // Learn Mode: Show reasoning/guidelines if present
    if (report.reasoning || report.guidelines) {
      html += `
      <div class="bg-[#1a2333] rounded-2xl p-6 md:p-8 shadow-inner border-2 border-[#138808]/50 mt-8 relative overflow-hidden">
        
        <!-- Decorative subtle grid -->
        <div class="absolute inset-0 opacity-20" style="background-image: radial-gradient(#4b5563 1px, transparent 1px); background-size: 16px 16px;"></div>
        
        <div class="relative z-10">
          <div class="flex items-center gap-3 mb-6 border-b border-slate-700 pb-4">
            <div class="bg-indigo-500/20 p-2 rounded-xl text-indigo-400">
               <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"></path></svg>
            </div>
            <div>
              <h3 class="text-xl font-bold text-white tracking-wide">Medical Student Learn Mode</h3>
              <p class="text-xs text-indigo-300 font-semibold uppercase tracking-widest mt-1">AI Reasoning Trace</p>
            </div>
          </div>
      `;
      
      if (report.reasoning) {
        html += `
        <div class="mb-6">
          <h4 class="text-sm font-bold text-indigo-400 mb-3 uppercase tracking-wider">Step-by-step Reasoning</h4>
          <div class="space-y-3">
        `;
        report.reasoning.forEach((r, idx) => { 
          html += `
            <div class="flex items-start gap-3 bg-slate-800/50 p-3 rounded-lg border border-slate-700/50">
              <span class="bg-indigo-900 text-indigo-300 text-xs font-bold w-6 h-6 rounded-full flex items-center justify-center shrink-0 mt-0.5">${idx+1}</span>
              <p class="text-slate-300 font-mono text-sm leading-relaxed">${r}</p>
            </div>
          `; 
        });
        html += `</div></div>`;
      }
      
      if (report.guidelines) {
        html += `
        <div>
          <h4 class="text-sm font-bold text-[#138808] mb-3 uppercase tracking-wider">Guideline Constraints Context</h4>
          <ul class="space-y-2">
        `;
        report.guidelines.forEach(g => { 
          html += `
            <li class="flex items-start gap-2 text-slate-400 font-mono text-sm bg-slate-800/30 p-2 rounded border-l-2 border-[#138808]">
              <span>${g}</span>
            </li>
          `; 
        });
        html += `</ul></div>`;
      }
      html += `</div></div>`;
    }

    reportContent.innerHTML = html;
    reportError.classList.add('hidden');
    
    // Advanced section (hidden by default) - Dark Theme
    let adv = '';
    adv += `<div class="mb-4 bg-slate-800 p-4 rounded-xl border border-slate-700">`;
    adv += `  <div class='mb-1 text-xs text-slate-500 uppercase font-bold tracking-widest'>Computed Bias Risk Score</div>`;
    adv += `  <div class='text-2xl font-black ${report.equity_check.bias_score > 0.5 ? 'text-red-500' : 'text-green-500'}'>${report.equity_check.bias_score}</div>`;
    adv += `</div>`;
    
    if (report.equity_check.potential_biases && report.equity_check.potential_biases.length) {
      adv += `<div class='mb-6'>`;
      adv += `  <div class='mb-2 text-sm font-bold text-orange-400 uppercase tracking-wider'>Identified Potential Biases</div>`;
      adv += `  <ul class='space-y-2'>`;
      report.equity_check.potential_biases.forEach(bias => {
        adv += `<li class="bg-slate-800/50 p-3 rounded border-l-4 border-orange-500 text-slate-300 font-mono text-sm">${bias}</li>`;
      });
      adv += `  </ul>`;
      adv += `</div>`;
    }
    
    if (report.equity_check.cultural_adaptations && report.equity_check.cultural_adaptations.length) {
      adv += `<div class='mb-6'>`;
      adv += `  <div class='mb-2 text-sm font-bold text-blue-400 uppercase tracking-wider'>Suggested Cultural Adaptations</div>`;
      adv += `  <ul class='space-y-2'>`;
      report.equity_check.cultural_adaptations.forEach(adapt => {
        adv += `<li class="bg-slate-800/50 p-3 rounded border-l-4 border-blue-500 text-slate-300 font-mono text-sm">${adapt}</li>`;
      });
      adv += `  </ul>`;
      adv += `</div>`;
    }
    adv += `<div class='mt-8 pt-6 border-t border-slate-700'>`;
    adv += `  <div class='mb-2 text-xs font-bold text-slate-500 uppercase tracking-widest'>Raw Debug Telemetry</div>`;
    adv += `  <pre class='bg-black/50 border border-slate-800 p-4 rounded-xl text-xs text-green-400 overflow-x-auto font-mono'>${JSON.stringify(report.debug_info, null, 2)}</pre>`;
    adv += `</div>`;
    
    advancedContent.innerHTML = adv;
    advancedSection.classList.add('hidden');
    if (toggleAdvancedBtn) toggleAdvancedBtn.textContent = 'View Analytics & Trace';
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
