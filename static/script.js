// Tampilkan/sembunyikan Severity berdasarkan pilihan penyakit
const diseaseSelect = document.getElementById('Disease_Type');
const severityGroup = document.getElementById('severityGroup');
const severitySelect = document.getElementById('Severity');

diseaseSelect.addEventListener('change', () => {
    if (diseaseSelect.value === 'None') {
        severityGroup.style.display = 'none';
        severitySelect.value = 'Mild'; // reset ke default
    } else {
        severityGroup.style.display = 'flex';
    }
});

document.getElementById('dietForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // UI states
    const btnText = document.querySelector('.btn-text');
    const loader = document.querySelector('.loader');
    const resultSection = document.getElementById('resultSection');
    const submitBtn = document.querySelector('.btn-submit');
    
    btnText.textContent = "Menganalisis...";
    loader.classList.remove('hidden');
    submitBtn.disabled = true;
    resultSection.classList.add('hidden');

    // Collect data
    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());
    
    // Parse numbers properly
    data.Age = parseInt(data.Age);
    data.Weight_kg = parseFloat(data.Weight_kg);
    data.Height_cm = parseFloat(data.Height_cm);
    data.Blood_Sugar_mgdL = parseFloat(data.Blood_Sugar_mgdL);
    data.Blood_Pressure_Systolic = parseInt(data.Blood_Pressure_Systolic);
    data.Cholesterol_mgdL = parseFloat(data.Cholesterol_mgdL);
    data.Weekly_Exercise_Hours = 3.0; // Default: tidak ditampilkan di UI (sudah direpresentasikan oleh Activity_Level)
    data.Daily_Caloric_Intake = parseFloat(data.Daily_Caloric_Intake);
    data.Adherence_to_Diet_Plan = parseFloat(data.Adherence_to_Diet_Plan);
    data.top_n_menus = parseInt(data.top_n_menus);
    // String fields — sudah terambil otomatis, tidak perlu di-parse
    // data.Severity, data.Dietary_Restrictions, data.Preferred_Cuisine

    try {
        const response = await fetch('/recommend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        displayResults(result);

    } catch (error) {
        console.error('Error:', error);
        alert('Terjadi kesalahan saat memproses data. Pastikan API server (app.py) sedang berjalan.');
    } finally {
        btnText.textContent = "Dapatkan Rekomendasi";
        loader.classList.add('hidden');
        submitBtn.disabled = false;
    }
});

function displayResults(data) {
    const resultSection = document.getElementById('resultSection');
    
    // Summary
    document.getElementById('dietType').textContent = data.diet_type.replace('_', ' ');
    const prob = data.diet_probability[data.diet_type] * 100;
    document.getElementById('dietProb').textContent = `${prob.toFixed(1)}% Keyakinan Model`;
    document.getElementById('targetKcal').textContent = data.target_calories_kcal;

    // Compliance Status
    const complianceDiv = document.getElementById('complianceStatus');
    complianceDiv.textContent = data.compliance_status;
    complianceDiv.className = 'compliance-status ' + 
        (data.compliance_status.startsWith('COMPLIANT') ? 'status-compliant' : 
         data.compliance_status.startsWith('PARTIAL') ? 'status-partial' : 'status-noncompliant');

    // Global Explanations
    const expList = document.getElementById('globalExpList');
    expList.innerHTML = '';
    data.global_explanation.forEach(exp => {
        const li = document.createElement('li');
        li.textContent = exp;
        expList.appendChild(li);
    });

    // Menus
    const menusContainer = document.getElementById('menusContainer');
    menusContainer.innerHTML = '';

    if (data.recommended_menus.length === 0) {
         menusContainer.innerHTML = `<div class="glass-panel" style="text-align:center; color: var(--warning);">
            <h3>Tidak ada kombinasi menu yang valid</h3>
            <p>Sistem tidak dapat menemukan kombinasi menu yang memenuhi semua batasan hard-constraint nutrisi untuk profil ini.</p>
         </div>`;
    } else {
        data.recommended_menus.forEach((menu, idx) => {
            const card = document.createElement('div');
            card.className = 'menu-card';
            
            // Menu Items HTML
            let itemsHtml = menu.menu_items.map(item => `<div class="menu-item">${item}</div>`).join('');
            
            // Explanations HTML
            let expHtml = menu.explanations.map(e => `<li>${e}</li>`).join('');

            card.innerHTML = `
                <div class="menu-header">
                    <h3>Pilihan Menu #${idx + 1}</h3>
                    <span class="menu-score">Skor: ${(menu.score * 100).toFixed(1)}</span>
                </div>
                
                <div class="menu-items">
                    ${itemsHtml}
                </div>

                <div class="nutrition-grid">
                    <div class="nut-item">
                        <span class="nut-label">Kalori</span>
                        <span class="nut-val">${menu.nutrition_summary.total_calories_kcal}</span>
                    </div>
                    <div class="nut-item">
                        <span class="nut-label">Karbo</span>
                        <span class="nut-val">${menu.nutrition_summary.total_carbs_g}g</span>
                    </div>
                    <div class="nut-item">
                        <span class="nut-label">Protein</span>
                        <span class="nut-val">${menu.nutrition_summary.total_protein_g}g</span>
                    </div>
                    <div class="nut-item">
                        <span class="nut-label">Lemak</span>
                        <span class="nut-val">${menu.nutrition_summary.total_fat_g}g</span>
                    </div>
                    <div class="nut-item">
                        <span class="nut-label">Sodium</span>
                        <span class="nut-val">${menu.nutrition_summary.total_sodium_mg}mg</span>
                    </div>
                </div>

                <div class="menu-explanations">
                    <ul>${expHtml}</ul>
                </div>
            `;
            menusContainer.appendChild(card);
        });
    }

    resultSection.classList.remove('hidden');
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
