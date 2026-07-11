import fs from "node:fs/promises";
import path from "node:path";
import { Workbook, SpreadsheetFile } from "@oai/artifact-tool";

const sourceDir = "D:/AAAProject/base_experi/PhysHSI/output_dir/logs/boxperturb_compare/model55500_vs_carrybox";
const outputDir = "D:/AAAProject/base_experi/PhysHSI/outputs/boxperturb_model_comparison";

function parseCsv(text) {
  const rows = [];
  let row = [], cell = "", quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') { cell += '"'; i++; }
      else if (c === '"') quoted = false;
      else cell += c;
    } else if (c === '"') quoted = true;
    else if (c === ',') { row.push(cell); cell = ""; }
    else if (c === '\n') { row.push(cell.replace(/\r$/, "")); rows.push(row); row = []; cell = ""; }
    else cell += c;
  }
  if (cell.length || row.length) { row.push(cell); rows.push(row); }
  const headers = rows[0];
  return rows.slice(1).filter(r => r.some(v => v !== "")).map(r => {
    const out = {};
    headers.forEach((h, i) => {
      const raw = r[i] ?? "";
      if (raw === "" || raw === "nan" || raw === "NaN") out[h] = null;
      else if (/^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/.test(raw)) out[h] = Number(raw);
      else out[h] = raw;
    });
    return out;
  });
}

const [trials, comparisons] = await Promise.all([
  fs.readFile(path.join(sourceDir, "trials.csv"), "utf8").then(parseCsv),
  fs.readFile(path.join(sourceDir, "comparison.csv"), "utf8").then(parseCsv),
]);

const models = [...new Set(trials.map(r => r.model))];
const baseline = "carrybox_builtin";
const interaction = "critic143_model55500";
const directions = ["+box_x", "-box_x", "+box_y", "-box_y", "+z_world", "-z_world"];
const betas = [0.10, 0.25, 0.50, 0.75];

const finite = values => values.filter(v => typeof v === "number" && Number.isFinite(v));
const avg = values => { const x = finite(values); return x.length ? x.reduce((a,b)=>a+b,0)/x.length : null; };
const count = (rows, predicate) => rows.filter(predicate).length;
const subset = (model, extra = () => true) => trials.filter(r => r.model === model && extra(r));
const eventRows = model => subset(model, r => r.event_triggered === 1);
const metric = (model, key, onlyEvents = true) => avg((onlyEvents ? eventRows(model) : subset(model)).map(r => r[key]));
const rate = (model, predicate, onlyEvents = false) => {
  const rows = onlyEvents ? eventRows(model) : subset(model);
  return rows.length ? count(rows, predicate) / rows.length : null;
};
const delta = (a, b) => a == null || b == null ? null : b - a;
const pctDelta = (a, b) => a ? (b - a) / Math.abs(a) : null;

const overallDefs = [
  ["前置条件成功率", m => rate(m, r => r.precondition_success === 1), "higher", "全部120 trials"],
  ["事件后终止率", m => rate(m, r => r.termination === 1, true), "lower", "仅event_triggered"],
  ["恢复成功率", m => metric(m, "recovery_success"), "higher", "仅event_triggered"],
  ["Pulse双手接触保持率", m => metric(m, "pulse_bimanual_contact_retention"), "higher", "仅event_triggered"],
  ["Post confirmed比例", m => metric(m, "post_confirmed_ratio"), "higher", "仅event_triggered"],
  ["最大Pitch (rad)", m => metric(m, "max_abs_pitch_rad"), "lower", "仅event_triggered"],
  ["手-箱相对速度RMS (m/s)", m => metric(m, "hand_box_rel_speed_rms_mps"), "lower", "仅event_triggered"],
  ["Force有效样本比例", m => metric(m, "force_valid_fraction"), "higher", "仅event_triggered"],
  ["Pulse有效样本比例", m => metric(m, "pulse_force_valid_fraction"), "higher", "仅event_triggered"],
  ["Closure residual", m => metric(m, "force_closure_residual_pulse_mean"), "lower", "仅event_triggered"],
  ["法向负载不对称度", m => metric(m, "normal_load_asymmetry_pulse_mean"), "lower", "仅event_triggered"],
];
const overallRows = overallDefs.map(([name, fn, better, scope]) => {
  const a = fn(baseline), b = fn(interaction);
  return [name, a, b, delta(a,b), pctDelta(a,b), better, scope];
});

function grouped(rows, key, keyValue, metricKey, eventOnly=true) {
  const x = rows.filter(r => r[key] === keyValue && (!eventOnly || r.event_triggered === 1));
  return avg(x.map(r => r[metricKey]));
}

const directionRows = directions.map(dir => [
  dir,
  grouped(subset(baseline), "direction", dir, "post_confirmed_ratio"),
  grouped(subset(interaction), "direction", dir, "post_confirmed_ratio"),
  grouped(subset(baseline), "direction", dir, "max_abs_pitch_rad"),
  grouped(subset(interaction), "direction", dir, "max_abs_pitch_rad"),
  grouped(subset(baseline), "direction", dir, "force_valid_fraction"),
  grouped(subset(interaction), "direction", dir, "force_valid_fraction"),
  grouped(subset(baseline), "direction", dir, "force_closure_residual_pulse_mean"),
  grouped(subset(interaction), "direction", dir, "force_closure_residual_pulse_mean"),
]);

const betaRows = betas.map(beta => [
  beta,
  grouped(subset(baseline), "requested_beta", beta, "termination"),
  grouped(subset(interaction), "requested_beta", beta, "termination"),
  grouped(subset(baseline), "requested_beta", beta, "post_confirmed_ratio"),
  grouped(subset(interaction), "requested_beta", beta, "post_confirmed_ratio"),
  grouped(subset(baseline), "requested_beta", beta, "max_abs_pitch_rad"),
  grouped(subset(interaction), "requested_beta", beta, "max_abs_pitch_rad"),
  grouped(subset(baseline), "requested_beta", beta, "force_valid_fraction"),
  grouped(subset(interaction), "requested_beta", beta, "force_valid_fraction"),
]);

const forceRows = directions.map(dir => [
  dir,
  grouped(subset(baseline), "direction", dir, "left_fn_raw_N_pulse_mean"),
  grouped(subset(interaction), "direction", dir, "left_fn_raw_N_pulse_mean"),
  grouped(subset(baseline), "direction", dir, "right_fn_raw_N_pulse_mean"),
  grouped(subset(interaction), "direction", dir, "right_fn_raw_N_pulse_mean"),
  grouped(subset(baseline), "direction", dir, "left_ft_raw_N_pulse_mean"),
  grouped(subset(interaction), "direction", dir, "left_ft_raw_N_pulse_mean"),
  grouped(subset(baseline), "direction", dir, "right_ft_raw_N_pulse_mean"),
  grouped(subset(interaction), "direction", dir, "right_ft_raw_N_pulse_mean"),
]);

const evidenceDefs = [
  ["Post confirmed比例", "post_confirmed_ratio_paired_mean_difference", "higher", "扰动后保持"],
  ["最大Pitch", "max_abs_pitch_rad_paired_mean_difference", "lower", "姿态稳定"],
  ["相对速度RMS", "hand_box_rel_speed_rms_mps_paired_mean_difference", "lower", "手箱滑移代理"],
  ["Force有效比例", "force_valid_fraction_paired_mean_difference", "higher", "分解可信度"],
  ["Closure residual", "force_closure_residual_pulse_mean_paired_mean_difference", "lower", "pair dominance"],
  ["左手Fn", "left_fn_raw_N_pulse_mean_paired_mean_difference", "diagnostic", "法向压紧"],
  ["右手Fn", "right_fn_raw_N_pulse_mean_paired_mean_difference", "diagnostic", "法向压紧"],
  ["左手Ft", "left_ft_raw_N_pulse_mean_paired_mean_difference", "diagnostic", "切向合力"],
  ["右手Ft", "right_ft_raw_N_pulse_mean_paired_mean_difference", "diagnostic", "切向合力"],
];
const evidenceRows = evidenceDefs.map(([label, key, preference, meaning]) => {
  const vals = finite(comparisons.map(r => r[key]));
  const mean = avg(vals);
  let favorable = null;
  if (preference === "higher") favorable = vals.filter(v => v > 0).length / vals.length;
  if (preference === "lower") favorable = vals.filter(v => v < 0).length / vals.length;
  return [label, mean, favorable, vals.length, preference, meaning];
});

const workbook = Workbook.create();
const dashboard = workbook.worksheets.add("Dashboard");
const byDirection = workbook.worksheets.add("By Direction");
const byBeta = workbook.worksheets.add("By Beta");
const forceSheet = workbook.worksheets.add("Force Analysis");
const pairedSheet = workbook.worksheets.add("Paired Cells");
const trialSheet = workbook.worksheets.add("Trial Data");
const methodSheet = workbook.worksheets.add("Method");

const navy = "#17324D", blue = "#3977B8", orange = "#E07A3F", light = "#EAF1F7", pale = "#F5F8FB";
function title(sheet, text, subtitle) {
  sheet.showGridLines = false;
  sheet.getRange("A1:H1").merge(); sheet.getRange("A1").values = [[text]];
  sheet.getRange("A1:H1").format = { fill: navy, font: { bold: true, color: "#FFFFFF", size: 18 }, rowHeight: 30 };
  sheet.getRange("A2:H2").merge(); sheet.getRange("A2").values = [[subtitle]];
  sheet.getRange("A2:H2").format = { fill: light, font: { color: navy, italic: true }, wrapText: true, rowHeight: 34 };
}
function header(range) {
  range.format = { fill: navy, font: { bold: true, color: "#FFFFFF" }, wrapText: true, borders: { preset: "outside", style: "thin", color: "#9FB3C8" } };
}
function styleTable(sheet, range, numRange=null) {
  range.format.borders = { preset: "inside", style: "thin", color: "#D9E2EA" };
  range.format.rowHeight = 20;
  if (numRange) sheet.getRange(numRange).format.numberFormat = "0.000";
}

title(dashboard, "CarryBox Box-Perturbation 模型对比", "critic143_model55500 与 carrybox_builtin；同方向、beta、seed 配对。描述性结论，5 seeds 不支持强显著性声明。");
dashboard.getRange("A4:G4").values = [["指标", "carrybox_builtin", "model55500", "差值", "相对变化", "更优方向", "统计范围"]];
dashboard.getRange(`A5:G${4+overallRows.length}`).values = overallRows;
header(dashboard.getRange("A4:G4")); styleTable(dashboard, dashboard.getRange(`A4:G${4+overallRows.length}`));
dashboard.getRange(`B5:D${4+overallRows.length}`).format.numberFormat = "0.000";
dashboard.getRange(`E5:E${4+overallRows.length}`).format.numberFormat = "0.0%";
dashboard.getRange("A18:G18").merge(); dashboard.getRange("A18").values = [["核心判读"]];
dashboard.getRange("A18:G18").format = { fill: blue, font: { bold: true, color: "#FFFFFF" } };
dashboard.getRange("A19:G24").merge(true);
dashboard.getRange("A19:A24").values = [
  ["1. model55500 的前置成功率为 100%，builtin 为 80%；builtin 的 seed=2 在全部24个cell均未进入扰动阶段。"],
  ["2. 在成功进入扰动的 trials 中，model55500 的 Post-confirmed 接近 100%，最大Pitch和closure residual明显降低。"],
  ["3. model55500 的事件后终止率并未降低（4.17% vs 2.08%）；优势主要是前置可达性与扰动后持续保持。"],
  ["4. model55500 的手-箱相对速度RMS更高，说明接触稳定提升并不等价于更低相对运动，需要继续审查滑移。"],
  ["5. 左手Fn更高、左手Ft更低，但右手Ft略高；结合closure与validity改善，趋势支持更强压紧，但不等于真实pairwise摩擦力。"],
  ["6. raw rho 在Fn接近0的无效样本中会爆炸，不作为模型优劣结论；应优先看valid fraction、closure和Fn/Ft。"],
];
dashboard.getRange("A19:G24").format = { fill: pale, wrapText: true, rowHeight: 28, font: { color: "#243746" } };
dashboard.getRange("A:A").format.columnWidth = 31; dashboard.getRange("B:F").format.columnWidth = 15; dashboard.getRange("G:G").format.columnWidth = 20;
dashboard.freezePanes.freezeRows(4);

const robustnessChartData = [["指标", "carrybox_builtin", "model55500"], ...overallRows.slice(0,5).map(r => [r[0], r[1], r[2]])];
dashboard.getRange("J2:L7").values = robustnessChartData;
dashboard.getRange("J:J").format.columnWidth=28; dashboard.getRange("K:L").format.columnWidth=16; dashboard.getRange("K3:L7").format.numberFormat="0.0%";
const chart1 = dashboard.charts.add("bar", dashboard.getRange("J2:L7")); chart1.title = "任务鲁棒性与保持率"; chart1.hasLegend = true; chart1.yAxis = { numberFormatCode: "0%", min: 0, max: 1 }; chart1.setPosition("I4", "P16");
const stabilityData = [["指标", "carrybox_builtin", "model55500"], ...overallRows.slice(5,7).map(r => [r[0], r[1], r[2]])];
dashboard.getRange("J19:L21").values = stabilityData;
const chart2 = dashboard.charts.add("bar", dashboard.getRange("J19:L21")); chart2.title = "姿态与相对运动（越低越好）"; chart2.hasLegend = true; chart2.setPosition("I18", "P30");

title(byDirection, "按扰动方向比较", "所有 beta 与有效 seeds 聚合；事件后指标仅使用 event_triggered=1 trials。");
const dirHeaders = ["方向","Post builtin","Post 55500","Pitch builtin","Pitch 55500","Valid builtin","Valid 55500","Closure builtin","Closure 55500"];
byDirection.getRange("A4:I4").values=[dirHeaders]; byDirection.getRange("A5:I10").values=directionRows; header(byDirection.getRange("A4:I4")); styleTable(byDirection,byDirection.getRange("A4:I10"));
byDirection.getRange("B5:C10").format.numberFormat="0.0%"; byDirection.getRange("D5:I10").format.numberFormat="0.000"; byDirection.getRange("A:I").format.columnWidth=16;
const d1=byDirection.charts.add("bar",byDirection.getRange("A4:C10")); d1.title="Post-confirmed by Direction"; d1.yAxis={numberFormatCode:"0%",min:0,max:1}; d1.setPosition("A13","H29");
// Use compact helper tables to avoid non-contiguous chart sources.
byDirection.getRange("K4:M10").values=[["方向","Pitch builtin","Pitch 55500"],...directionRows.map(r=>[r[0],r[3],r[4]])];
byDirection.getRange("K:M").format.columnWidth=16; byDirection.getRange("L5:M28").format.numberFormat="0.000";
const d3=byDirection.charts.add("bar",byDirection.getRange("K4:M10")); d3.title="Max Pitch by Direction (rad)"; d3.setPosition("I13","P29");
byDirection.getRange("K13:M19").values=[["方向","Valid builtin","Valid 55500"],...directionRows.map(r=>[r[0],r[5],r[6]])];
const d4=byDirection.charts.add("bar",byDirection.getRange("K13:M19")); d4.title="Force-valid Fraction"; d4.yAxis={numberFormatCode:"0%",min:0,max:1}; d4.setPosition("A31","H47");
byDirection.getRange("K22:M28").values=[["方向","Closure builtin","Closure 55500"],...directionRows.map(r=>[r[0],r[7],r[8]])];
const d5=byDirection.charts.add("bar",byDirection.getRange("K22:M28")); d5.title="Closure Residual (lower is better)"; d5.setPosition("I31","P47");
byDirection.freezePanes.freezeRows(4);

title(byBeta, "按扰动力等级比较", "四个 beta 聚合六方向；终止率为进入扰动后的条件终止率。");
const betaHeaders=["beta","Term builtin","Term 55500","Post builtin","Post 55500","Pitch builtin","Pitch 55500","Valid builtin","Valid 55500"];
byBeta.getRange("A4:I4").values=[betaHeaders]; byBeta.getRange("A5:I8").values=betaRows; header(byBeta.getRange("A4:I4")); styleTable(byBeta,byBeta.getRange("A4:I8"));
byBeta.getRange("A5:A8").format.numberFormat="0.00"; byBeta.getRange("B5:E8").format.numberFormat="0.0%"; byBeta.getRange("F5:I8").format.numberFormat="0.000"; byBeta.getRange("A:I").format.columnWidth=16;
byBeta.getRange("K4:M8").values=[["beta","Post builtin","Post 55500"],...betaRows.map(r=>[r[0],r[3],r[4]])];
byBeta.getRange("K:M").format.columnWidth=16;
const b1=byBeta.charts.add("line",byBeta.getRange("K4:M8")); b1.title="Post-confirmed vs Beta"; b1.yAxis={numberFormatCode:"0%",min:0,max:1}; b1.setPosition("A11","H27");
byBeta.getRange("K11:M15").values=[["beta","Pitch builtin","Pitch 55500"],...betaRows.map(r=>[r[0],r[5],r[6]])];
const b2=byBeta.charts.add("line",byBeta.getRange("K11:M15")); b2.title="Max Pitch vs Beta (rad)"; b2.setPosition("I11","P27");
byBeta.getRange("K18:M22").values=[["beta","Valid builtin","Valid 55500"],...betaRows.map(r=>[r[0],r[7],r[8]])];
const b3=byBeta.charts.add("line",byBeta.getRange("K18:M22")); b3.title="Force-valid Fraction vs Beta"; b3.yAxis={numberFormatCode:"0%",min:0,max:1}; b3.setPosition("A29","H45");
byBeta.getRange("K25:M29").values=[["beta","Term builtin","Term 55500"],...betaRows.map(r=>[r[0],r[1],r[2]])];
const b4=byBeta.charts.add("line",byBeta.getRange("K25:M29")); b4.title="Post-event Termination vs Beta"; b4.yAxis={numberFormatCode:"0%",min:0}; b4.setPosition("I29","P45");
byBeta.freezePanes.freezeRows(4);

title(forceSheet, "Fn/Ft 响应比较", "Hand rigid-body net contact force 在锁定 box-face normal 上的投影；不是严格 pairwise hand-box 摩擦力。");
const forceHeaders=["方向","L Fn builtin","L Fn 55500","R Fn builtin","R Fn 55500","L Ft builtin","L Ft 55500","R Ft builtin","R Ft 55500"];
forceSheet.getRange("A4:I4").values=[forceHeaders]; forceSheet.getRange("A5:I10").values=forceRows; header(forceSheet.getRange("A4:I4")); styleTable(forceSheet,forceSheet.getRange("A4:I10")); forceSheet.getRange("B5:I10").format.numberFormat="0.00"; forceSheet.getRange("A:I").format.columnWidth=16;
forceSheet.getRange("K4:M10").values=[["方向","L Fn builtin","L Fn 55500"],...forceRows.map(r=>[r[0],r[1],r[2]])];
forceSheet.getRange("K:M").format.columnWidth=16; forceSheet.getRange("L5:M37").format.numberFormat="0.00";
const f1=forceSheet.charts.add("bar",forceSheet.getRange("K4:M10")); f1.title="Left Normal Force Fn (N)"; f1.setPosition("A13","H29");
forceSheet.getRange("K13:M19").values=[["方向","R Fn builtin","R Fn 55500"],...forceRows.map(r=>[r[0],r[3],r[4]])];
const f2=forceSheet.charts.add("bar",forceSheet.getRange("K13:M19")); f2.title="Right Normal Force Fn (N)"; f2.setPosition("I13","P29");
forceSheet.getRange("K22:M28").values=[["方向","L Ft builtin","L Ft 55500"],...forceRows.map(r=>[r[0],r[5],r[6]])];
const f3=forceSheet.charts.add("bar",forceSheet.getRange("K22:M28")); f3.title="Left Tangential Force Ft (N)"; f3.setPosition("A31","H47");
forceSheet.getRange("K31:M37").values=[["方向","R Ft builtin","R Ft 55500"],...forceRows.map(r=>[r[0],r[7],r[8]])];
const f4=forceSheet.charts.add("bar",forceSheet.getRange("K31:M37")); f4.title="Right Tangential Force Ft (N)"; f4.setPosition("I31","P47");
forceSheet.getRange("A50:I52").merge(true); forceSheet.getRange("A50:A52").values=[
  ["可信度限制：builtin 的 force_baseline_unavailable 为55%，model55500为19.2%。"],
  ["raw rho 在 Fn≈0 时数值爆炸，未用于图表结论。"],
  ["Fn/Ft 趋势必须与 closure residual≤0.2、双手接触及 sign verified 联合解读。"],
]; forceSheet.getRange("A50:I52").format={fill:"#FFF4E8",wrapText:true,font:{color:"#7A3E00"},rowHeight:25};
forceSheet.freezePanes.freezeRows(4);

title(pairedSheet, "24个方向×beta配对单元", "差值均为 model55500 − carrybox_builtin；每个cell只使用两模型都通过前置条件的相同seed。多数cell仅4个paired seeds。");
pairedSheet.getRange("A4:F4").values=[["指标","24-cell平均差值","有利方向cell比例","cell数","判读方向","物理含义"]];
pairedSheet.getRange(`A5:F${4+evidenceRows.length}`).values=evidenceRows; header(pairedSheet.getRange("A4:F4")); styleTable(pairedSheet,pairedSheet.getRange(`A4:F${4+evidenceRows.length}`)); pairedSheet.getRange(`B5:B${4+evidenceRows.length}`).format.numberFormat="0.000"; pairedSheet.getRange(`C5:C${4+evidenceRows.length}`).format.numberFormat="0.0%";
pairedSheet.getRange(`C5:C${4+evidenceRows.length}`).conditionalFormats.add("colorScale",{thresholds:["min","50%","max"],colors:["#F4A6A6","#FFF2CC","#A9D18E"]});
const compHeaders=Object.keys(comparisons[0]); const compMatrix=[compHeaders,...comparisons.map(r=>compHeaders.map(h=>r[h]))];
pairedSheet.getRangeByIndexes(18,0,compMatrix.length,compHeaders.length).values=compMatrix; header(pairedSheet.getRangeByIndexes(18,0,1,compHeaders.length)); pairedSheet.freezePanes.freezeRows(4); pairedSheet.getRange("A:F").format.columnWidth=20;
pairedSheet.getRangeByIndexes(18,0,1,compHeaders.length).format.wrapText=true; pairedSheet.getRangeByIndexes(18,0,1,compHeaders.length).format.rowHeight=52;

const trialHeaders=Object.keys(trials[0]); const trialMatrix=[trialHeaders,...trials.map(r=>trialHeaders.map(h=>r[h]))];
trialSheet.showGridLines=false; trialSheet.getRangeByIndexes(0,0,trialMatrix.length,trialHeaders.length).values=trialMatrix; header(trialSheet.getRangeByIndexes(0,0,1,trialHeaders.length)); trialSheet.freezePanes.freezeRows(1); trialSheet.freezePanes.freezeColumns(5); trialSheet.getUsedRange().format.rowHeight=18;
trialSheet.getRange("A:A").format.columnWidth=24; trialSheet.getRange("B:B").format.columnWidth=52; trialSheet.getRange("C:C").format.columnWidth=8; trialSheet.getRange("D:D").format.columnWidth=13; trialSheet.getRange("E:E").format.columnWidth=11;
trialSheet.getRange("F:M").format.columnWidth=16; trialSheet.getRange("N:N").format.columnWidth=18; trialSheet.getRange("O:P").format.columnWidth=12; trialSheet.getRangeByIndexes(0,0,1,trialHeaders.length).format.wrapText=true; trialSheet.getRangeByIndexes(0,0,1,trialHeaders.length).format.rowHeight=48;

title(methodSheet, "方法与限制", "用于复现实验判读和避免错误归因。");
methodSheet.getRange("A4:B12").values=[
  ["项目","说明"],
  ["模型",`${baseline} vs ${interaction}`],
  ["样本","每模型120 trials：6方向×4 beta×5 seeds"],
  ["配对规则","相同 direction、beta、seed；仅两模型均precondition_success的trial进入paired difference"],
  ["有效paired seeds","builtin的seed=2全部前置失败，因此大多数cell只有4个paired seeds"],
  ["Force定义","hand rigid-body net contact force对估计box-face normal的投影"],
  ["Force validity","confirmed carry、双手contact、normal sign verified、两侧Fn_signed>0、closure≤0.2"],
  ["统计立场","描述性对比；5 seeds不足以做强显著性声明，且同seed跨cell重复，不视为独立样本"],
  ["源目录",sourceDir],
]; header(methodSheet.getRange("A4:B4")); methodSheet.getRange("A5:B12").format={wrapText:true,fill:pale,borders:{preset:"inside",style:"thin",color:"#D9E2EA"},rowHeight:32}; methodSheet.getRange("A:A").format.columnWidth=24; methodSheet.getRange("B:B").format.columnWidth=95;

await fs.mkdir(outputDir,{recursive:true});
const xlsx=await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(path.join(outputDir,"boxperturb_model_comparison.xlsx"));
for (const [name, range] of [["Dashboard",null],["By Direction",null],["By Beta",null],["Force Analysis",null],["Paired Cells","A1:F28"],["Trial Data","A1:P15"],["Method","A1:H14"]]) {
  const options={sheetName:name,scale:1,format:"png"};
  if (range) options.range=range; else options.autoCrop="all";
  const preview=await workbook.render(options);
  await fs.writeFile(path.join(outputDir,`${name.replaceAll(" ","_")}.png`),new Uint8Array(await preview.arrayBuffer()));
}
const inspect=await workbook.inspect({kind:"table",range:"Dashboard!A1:G24",include:"values,formulas",tableMaxRows:24,tableMaxCols:7,maxChars:5000});
console.log(inspect.ndjson);
