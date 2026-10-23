/* Reproduce the revised USDA ComBase Predictor dynamic curves with its
 * downloaded, dated client-side code and model parameters. */
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const root = path.resolve(__dirname, '..', '..', '..');
const data=path.join(root,'06_Results','Figure_Source_Data','Official_Predictor');
for (const name of ['dmNumerics_official_20260926.js', 'cbMath_official_20260926.js']) {
  vm.runInThisContext(fs.readFileSync(path.join(root,'05_Data','Raw','Official_Predictor',name), 'utf8'), {filename: name});
}
const inputs = JSON.parse(fs.readFileSync(path.join(data, 'official_scenario_inputs.json'), 'utf8'));
const output = [];
for (const item of inputs) {
  const m = item.official_model;
  // Physiological state 1 is the website's no-lag setting, corresponding to h0=0.
  const result = combaseModels.getDynamicPredictionResult(
    0, item.initial_count, item.temp_profile, item.ph, item.aw,
    0, -35, m.ModelType, m.YMax, m.Coeff);
  const points = result.predPoints.map(p => ({time_h: p[0], predicted_log: p[1]}));
  if (!points.length || points.some(p => !Number.isFinite(p.predicted_log))) {
    throw new Error(`Invalid official curve for record ${item.record_id}`);
  }
  output.push({record_id: item.record_id, model_id: m.ID, points});
  console.log(item.record_id, points.length, points[points.length-1]);
}
fs.writeFileSync(path.join(data, 'official_curves_raw.json'), JSON.stringify(output));
