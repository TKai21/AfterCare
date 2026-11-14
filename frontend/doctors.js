// Shared patient-side helpers for the doctor choice: the doctor list, the doctor
// remembered in this browser, the direct-question request, and the emergency banner.
import {api, escapeHtml} from '/static/api.js';

const esc = escapeHtml;

const DOCTOR_KEY = 'aftercare.doctorId';

export const EMERGENCY_COPY = 'If you are having this symptom now, call 911 or follow the emergency instructions in your discharge document.';
export const MAX_QUESTION = 2000;

export const CHECK_ICON = '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M3.5 8.5l3 3 6-7" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
const ALERT_ICON = '<svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true"><path d="M12 3.5L2.5 20h19L12 3.5z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M12 10v4.5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><circle cx="12" cy="17.3" r="1.2" fill="currentColor"/></svg>';

let doctorsPromise = null;

// The doctors, ordered by id. A failed request is not cached.
export function loadDoctors() {
  if (!doctorsPromise) {
    doctorsPromise = api('/api/doctors')
      .then(data => (data && Array.isArray(data.doctors) ? data.doctors : []))
      .catch(error => {
        doctorsPromise = null;
        throw error;
      });
  }
  return doctorsPromise;
}

export function rememberDoctor(id) {
  try {
    localStorage.setItem(DOCTOR_KEY, String(id));
  } catch (error) {
    // Storage unavailable: the choice just is not remembered.
  }
}

// The remembered doctor if still in the list, otherwise the first doctor.
export function pickDoctor(doctors) {
  let saved = null;
  try {
    saved = localStorage.getItem(DOCTOR_KEY);
  } catch (error) {
    saved = null;
  }
  const found = doctors.find(d => String(d.id) === saved);
  return found || doctors[0] || null;
}

// POST /api/care-team -> {question, emergency: {terms, banner}}.
export function askDoctorDirectly(doctorId, question) {
  return api('/api/care-team', {method: 'POST', json: {doctor_id: doctorId, question}});
}

// signs: active warning-sign items of the discharge document, if known.
export function emergencyHtml(signs) {
  const list = Array.isArray(signs) ? signs : [];
  return `
    <section class="emergency" aria-label="Emergency">
      <span class="emergency-icon">${ALERT_ICON}</span>
      <div class="emergency-body">
        <p class="emergency-copy">${esc(EMERGENCY_COPY)}</p>
        ${list.length ? `
          <p class="emergency-label">From your discharge document</p>
          <ul class="emergency-list">${list.map(s => `<li>${esc(s.text)}</li>`).join('')}</ul>` : ''}
        <a class="btn btn-emergency" href="tel:911">Call 911</a>
      </div>
    </section>`;
}

export function warningSignsOf(doc) {
  if (!doc || !Array.isArray(doc.items)) return [];
  return doc.items.filter(i => i.category === 'warning_signs' && i.status === 'active');
}
