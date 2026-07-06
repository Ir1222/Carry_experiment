import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = "D:/AAAProject/base_experi/PhysHSI/output_dir/logs/boxperturb_ab";
const outDir = `${root}/audit_charts`;
const data = JSON.parse(await fs.readFile("D:/AAAProject/base_experi/PhysHSI/.codex_tmp_boxperturb_audit/audit_data.json", "utf8"));
await fs.mkdir(outDir, { recursive: true });

const wb = Workbook.create();
const navy = "#17365D", blue = "#4C78A8", red = "#E45756", pale = "#EAF2F8", gray = "#F3F4F6";
const titleFmt = { fill: navy, font: { bold: true, color: "#FFFFFF", size: 16 }, verticalAlignment: "center" };
const headFmt = { fill: "#D9EAF7", font: { bold: true, color: "#17365D" }, borders: { preset: "bottom", style: "thin", color: "#9CA3AF" }, wrapText: true };
const sectionFmt = { fill: navy, font: { bold: true, color: "#FFFFFF" } };

function writeTable(sheet, startRow, startCol, headers, rows, formats = {}) {
  const r = sheet.getRangeByIndexes(startRow, startCol, rows.length + 1, headers.length);
  r.values = [headers, ...rows];
  sheet.getRangeByIndexes(startRow, startCol, 1, headers.length).format = headFmt;
  if (formats.percentCols) for (const c of formats.percentCols) sheet.getRangeByIndexes(startRow + 1, startCol + c, rows.length, 1).format.numberFormat = "0.0%";
  if (formats.decimalCols) for (const c of formats.decimalCols) sheet.getRangeByIndexes(startRow + 1, startCol + c, rows.length, 1).format.numberFormat = "0.000";
  return r;
}

const dash = wb.worksheets.add("Dashboard");
dash.showGridLines = false;
dash.getRange("A1:P2").merge(); dash.getRange("A1").values = [["CarryBox Box-Perturbation A/B Audit"]]; dash.getRange("A1:P2").format = titleFmt;
dash.getRange("A3:P3").merge(); dash.getRange("A3").values = [["Builtin actor vs interaction-privileged-trained actor | 5 directions x 4 beta x 5 seeds"]]; dash.getRange("A3:P3").format = { fill: pale, font: { italic: true, color: navy } };

const overallHeaders = ["Model","Trials","Precondition","Triggered","Pulse hold","Bimanual","Recovery","Drop","Post confirmed","Rel RMS m/s","Rel peak m/s","Roll deg","Pitch deg","Hand asym","L hand N","R hand N","Resistive N"];
const overallRows = data.overall.map(d => [d.model,d.trials,d.precondition,d.triggered,d.pulse_hold,d.bimanual,d.recovery,d.drop,d.post_confirmed,d.rel_rms,d.rel_peak,d.roll_deg,d.pitch_deg,d.hand_asym,d.left_hand,d.right_hand,d.resistive]);
writeTable(dash,4,0,overallHeaders,overallRows,{percentCols:[2,4,5,6,7,8],decimalCols:[9,10,11,12,13,14,15,16]});
dash.getRange("A9:H9").merge(); dash.getRange("A9").values = [["Interpretation: interaction_priv holds contact better during the pulse, but has more later drops. Absolute tilt includes precondition, so it is not a pure perturbation delta."]]; dash.getRange("A9:H9").format = { fill: "#FFF4E5", font: { color: "#8A4B08" }, wrapText: true };

const rateData = [["Metric","Builtin","Interaction-priv"],
  ["Precondition",data.overall[0].precondition,data.overall[1].precondition],
  ["Pulse hold",data.overall[0].pulse_hold,data.overall[1].pulse_hold],
  ["Bimanual",data.overall[0].bimanual,data.overall[1].bimanual],
  ["Recovery",data.overall[0].recovery,data.overall[1].recovery],
  ["No drop",1-data.overall[0].drop,1-data.overall[1].drop],
  ["Post confirmed",data.overall[0].post_confirmed,data.overall[1].post_confirmed]];
dash.getRange("A12:C18").values = rateData; dash.getRange("A12:C12").format = headFmt; dash.getRange("B13:C18").format.numberFormat = "0.0%";
const c1 = dash.charts.add("bar", dash.getRange("A12:C18")); c1.title = "Overall retention and success rates"; c1.hasLegend = true; c1.yAxis = { numberFormatCode: "0%", min: 0, max: 1 }; c1.setPosition("E11","K27");

const betas=[0.1,0.25,0.5,0.75];
const getB=(m,b)=>data.beta.find(x=>x.model===m && x.beta===b);
const betaRate=[["beta","Builtin recovery","Interaction recovery","Builtin drop","Interaction drop"]];
for(const b of betas) betaRate.push([b,getB("builtin",b).recovery,getB("interaction_priv",b).recovery,getB("builtin",b).drop,getB("interaction_priv",b).drop]);
dash.getRange("A21:E25").values=betaRate; dash.getRange("A21:E21").format=headFmt; dash.getRange("B22:E25").format.numberFormat="0.0%";
const c2=dash.charts.add("line",dash.getRange("A21:E25")); c2.title="Recovery and drop versus beta"; c2.hasLegend=true; c2.yAxis={numberFormatCode:"0%",min:0,max:1}; c2.setPosition("L11","P27");

const betaMotion=[["beta","Builtin rel RMS","Interaction rel RMS","Builtin asym","Interaction asym"]];
for(const b of betas) betaMotion.push([b,getB("builtin",b).rel_rms,getB("interaction_priv",b).rel_rms,getB("builtin",b).hand_asym,getB("interaction_priv",b).hand_asym]);
dash.getRange("A29:E33").values=betaMotion; dash.getRange("A29:E29").format=headFmt; dash.getRange("B30:E33").format.numberFormat="0.000";
const c3=dash.charts.add("line",dash.getRange("A29:C33")); c3.title="Hand-box relative speed RMS (m/s)"; c3.hasLegend=true; c3.setPosition("E29","K45");
const c4=dash.charts.add("line",dash.getRange("A29:A33")); c4.setData(dash.getRange("A29:A33"));
// Use a compact contiguous helper range for asymmetry.
dash.getRange("G29:I33").values=[["beta","Builtin asym","Interaction asym"],...betas.map(b=>[b,getB("builtin",b).hand_asym,getB("interaction_priv",b).hand_asym])]; dash.getRange("G29:I29").format=headFmt;
c4.setData(dash.getRange("G29:I33")); c4.title="Hand load asymmetry proxy"; c4.hasLegend=true; c4.setPosition("L29","P45");

const seedRows=[["Model","Seed","Precondition","Valid","Drop","Recovery","Post confirmed"],...data.seed.map(d=>[d.model,d.seed,d.precondition,d.valid,d.drop,d.recovery,d.post_confirmed])];
dash.getRange("A48:G58").values=seedRows; dash.getRange("A48:G48").format=headFmt; dash.getRange("C49:C58").format.numberFormat="0.0%"; dash.getRange("E49:G58").format.numberFormat="0.0%";
dash.getRange("A60:P61").merge(); dash.getRange("A60").values=[["Seed concentration warning: interaction_priv drops are concentrated in seed 1 and seed 5; beta response is not monotonic. A beta=0 sham control is required before attributing these drops to the applied force."]]; dash.getRange("A60:P61").format={fill:"#FDECEC",font:{color:"#9B1C1C",bold:true},wrapText:true};
dash.freezePanes.freezeRows(4);
dash.getRange("A1:P61").format.autofitRows();
for (const col of ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P"]) dash.getRange(`${col}:${col}`).format.columnWidth = col==="A"?20:13;

const force = wb.worksheets.add("Force and Hand Proxy"); force.showGridLines=false;
force.getRange("A1:L2").merge(); force.getRange("A1").values=[["Applied Force and Hand-Contact Proxy Diagnostics"]]; force.getRange("A1:L2").format=titleFmt;
writeTable(force,3,0,["beta","Theory peak N","Actual peak N","Impulse Ns","Free-box delta-v m/s"],data.force.map(d=>[d.beta,d.theory_peak_N,d.actual_peak_N,d.impulse_Ns,d.free_delta_v_mps]),{decimalCols:[0,1,2,3,4]});
const profileMap=new Map(); for(const r of data.profiles){ if(!profileMap.has(r.time_s)) profileMap.set(r.time_s,{}); profileMap.get(r.time_s)[r.beta]=r.force_N; }
const profileRows=[["time s","beta 0.10","beta 0.25","beta 0.50","beta 0.75"],...Array.from(profileMap.entries()).map(([time,v])=>[time,v[0.1],v[0.25],v[0.5],v[0.75]])];
force.getRange("A11:E31").values=profileRows; force.getRange("A11:E11").format=headFmt; force.getRange("A12:E31").format.numberFormat="0.000";
const fc=force.charts.add("line",force.getRange("A11:E31")); fc.title="Actual midpoint-sampled 0.10 s half-sine pulse"; fc.hasLegend=true; fc.xAxis={axisType:"textAxis"}; fc.yAxis={numberFormatCode:"0.0"}; fc.setPosition("G4","L21");
const handRows=[["beta","Builtin L","Builtin R","Interaction L","Interaction R"],...betas.map(b=>[b,getB("builtin",b).left_hand,getB("builtin",b).right_hand,getB("interaction_priv",b).left_hand,getB("interaction_priv",b).right_hand])];
force.getRange("A34:E38").values=handRows; force.getRange("A34:E34").format=headFmt; force.getRange("B35:E38").format.numberFormat="0.00";
const hc=force.charts.add("bar",force.getRange("A34:E38")); hc.title="Mean hand net-contact proxy during pulse (N)"; hc.hasLegend=true; hc.setPosition("G23","L40");
force.getRange("A42:L45").merge(); force.getRange("A42").values=[["Important: hand_on_box_proxy = -hand rigid-body net-contact force. It is not a pairwise hand-box wrench. In this dataset, pairwise contact counts are zero while hand contact flags are active, so the pairwise audit failed and these values must remain explanatory proxies only."]]; force.getRange("A42:L45").format={fill:"#FFF4E5",font:{color:"#8A4B08",bold:true},wrapText:true};
force.getRange("A1:L45").format.autofitRows(); for(const col of ["A","B","C","D","E","F","G","H","I","J","K","L"]) force.getRange(`${col}:${col}`).format.columnWidth=16;

const trialDefs = [
["model","text","trial identity","Model label: builtin or interaction_priv."],["checkpoint","path","trial identity","Actor checkpoint path; critic is intentionally not loaded."],["seed","integer","trial identity","Requested Python/NumPy/Torch seed."],["direction","enum","input","Frozen perturbation direction: +/-box_x, +/-box_y, or -z_world."],["requested_beta","ratio","input","Requested force ratio beta in Fpeak=beta*m*g."],["precondition_success","0/1","gate","Reached 20 consecutive confirmed-carry policy steps within 5 s."],["event_triggered","0/1","gate","One perturbation pulse was scheduled and applied."],["peak_force_N","N","input","Theoretical capped peak force; NaN if gate failed."],["pulse_hold_retention","ratio","pulse","Fraction of policy steps during the 0.10 s pulse with confirmed_carry=true."],["pulse_bimanual_contact_retention","ratio","pulse","Fraction of pulse policy steps with both hand contact flags true."],["recovery_success","0/1","recovery","Obtained 5 consecutive confirmed-carry policy steps within 1 s after pulse."],["recovery_time_s","s","recovery","Time from pulse completion until the five-step recovery criterion is reached."],["termination","0/1","outcome","Trial reset by a retained physical failure condition."],["termination_reason","text","outcome","Recorded reason such as drop, box_tilt, head_low, base_low, base_tilt, timeout or other."],["drop_failure","0/1","outcome","termination_reason contains drop or box_tilt."],["fall_failure","0/1","outcome","termination_reason contains head_low, base_low or base_tilt."],["post_confirmed_ratio","ratio","post","Confirmed-carry fraction in the 2 s continuation window; becomes truncated/zero on early termination."],["goal_progress_m","m","post","Goal distance at pulse start minus goal distance at end; NaN for terminated trials."],["max_abs_roll_rad","rad","whole trial","Maximum absolute base roll across precondition, pulse, recovery and post windows."],["max_abs_pitch_rad","rad","whole trial","Maximum absolute base pitch across precondition, pulse, recovery and post windows."],["external_impulse_Ns","N*s","pulse","Sum of |Fext| over pulse physics substeps multiplied by sim.dt."],["left_hand_pre_mean_N","N","pre","Mean left hand net-contact proxy norm over the last 40 confirmed pre rows."],["right_hand_pre_mean_N","N","pre","Mean right hand net-contact proxy norm over the last 40 confirmed pre rows."],["left_hand_pulse_mean_N","N","pulse","Mean left hand proxy norm during pulse."],["right_hand_pulse_mean_N","N","pulse","Mean right hand proxy norm during pulse."],["left_hand_pulse_peak_N","N","pulse","Maximum left hand proxy norm during pulse."],["right_hand_pulse_peak_N","N","pulse","Maximum right hand proxy norm during pulse."],["left_hand_recovery_mean_N","N","recovery","Mean left hand proxy norm during recorded recovery rows."],["right_hand_recovery_mean_N","N","recovery","Mean right hand proxy norm during recorded recovery rows."],["left_hand_force_delta_N","N","derived","left_hand_pulse_mean_N - left_hand_pre_mean_N."],["right_hand_force_delta_N","N","derived","right_hand_pulse_mean_N - right_hand_pre_mean_N."],["resistive_hand_force_mean_N","N","pulse","Mean dot(Fleft_proxy+Fright_proxy, -perturb_direction). Positive opposes perturbation."],["resistive_hand_force_peak_N","N","pulse","Maximum signed resistive proxy during pulse."],["hand_load_asymmetry_mean","ratio","pulse","Mean |L-R|/(L+R), using hand proxy norms; 0 is balanced."],["hand_box_rel_speed_rms_mps","m/s","pulse+recovery","RMS of both hand-to-box linear relative-speed norms over pulse and recovery rows."],["hand_box_rel_speed_peak_mps","m/s","pulse+recovery","Maximum left or right hand-to-box relative-speed norm."],["pairwise_left_normal_mean_N","lambda proxy","pulse audit","Mean summed pairwise left-hand/box normal lambda; currently zero/unusable."],["pairwise_right_normal_mean_N","lambda proxy","pulse audit","Mean summed pairwise right-hand/box normal lambda; currently zero/unusable."],["pairwise_left_contact_count_mean","count","pulse audit","Mean matched left-hand/box contact count per pulse substep."],["pairwise_right_contact_count_mean","count","pulse audit","Mean matched right-hand/box contact count per pulse substep."],["pairwise_proxy_unmatched_fraction","ratio","pulse audit","Fraction where hand flag is active but no hand-box pair was found; 1.0 here means audit failure."],["trace_rows","count","trace","Total force_trace rows for this trial."],["pre_trace_rows","count","trace","Number of selected confirmed pre rows used by trial summary (normally 40)."],["pulse_trace_rows","count","trace","Physics trace rows during pulse (normally 20)."],["recovery_trace_rows","count","trace","Physics rows in recovery; normally 200, shorter on termination."],["post_trace_rows","count","trace","Physics rows in 2 s post window; normally 400, shorter/zero on termination."]];

const traceDefs = [
["model","text","identity","Model label."],["checkpoint","path","identity","Actor checkpoint path."],["seed","integer","identity","Requested seed."],["direction","enum","input","Requested frozen direction name."],["requested_beta","ratio","input","Requested beta from command grid."],["phase","enum","time","pre, pulse, recovery or post."],["frame","count","time","Global Isaac Gym simulation frame count."],["policy_step","count","time","Global policy/common step counter."],["physics_substep","0..decimation-1","time","Substep index inside current policy step."],["elapsed_pulse_physics_steps","count","time","Pulse substeps elapsed after force application."],["beta","ratio","state","Scheduled beta stored in environment; zero before scheduling."],["box_mass_kg","kg","state","Actual runtime box rigid-body mass."],["force_peak_N","N","input","Scheduled theoretical/capped peak force."],["confirmed_streak_at_schedule","policy steps","gate","Confirmed-carry streak when pulse was committed."],["f_ext_norm_N","N","input","Instantaneous magnitude of the applied half-sine external force."],["left_hand_on_box_proxy_norm_N","N","contact proxy","Norm of negative left-hand rigid-body net-contact force."],["right_hand_on_box_proxy_norm_N","N","contact proxy","Norm of negative right-hand rigid-body net-contact force."],["combined_hand_on_box_proxy_norm_N","N","contact proxy","Norm of vector sum of both hand-on-box proxies."],["box_net_contact_force_norm_N","N","contact","Norm of box rigid-body net-contact force, excluding directly applied force tensor."],["resistive_hand_force_N","N","derived","dot(combined hand proxy, -frozen perturbation direction)."],["hand_load_asymmetry","ratio","derived","|left norm-right norm|/(left norm+right norm)."],["left_hand_box_rel_speed_mps","m/s","motion","Norm of left-hand linear velocity minus box linear velocity."],["right_hand_box_rel_speed_mps","m/s","motion","Norm of right-hand linear velocity minus box linear velocity."],["left_contact","0/1","contact","Left proxy norm exceeds configured hand-contact-force threshold."],["right_contact","0/1","contact","Right proxy norm exceeds threshold."],["confirmed_carry","0/1","task state","Environment confirmed_carry_buf at this substep sample."],["box_lin_speed_mps","m/s","motion","Norm of box world linear velocity."],["box_ang_speed_radps","rad/s","motion","Norm of box world angular velocity."],["f_ext_world_N_x/y/z","N","input vector","World/ENV-space components of applied external force."],["left_hand_on_box_proxy_world_N_x/y/z","N","contact vector","World components of negative left-hand net-contact force."],["right_hand_on_box_proxy_world_N_x/y/z","N","contact vector","World components of negative right-hand net-contact force."],["box_net_contact_force_world_N_x/y/z","N","contact vector","World components of box net-contact force."],["box_lin_vel_world_mps_x/y/z","m/s","motion vector","World components of box linear velocity."],["box_ang_vel_world_radps_x/y/z","rad/s","motion vector","World components of box angular velocity."],["left_pair_count","count","pair audit","Matched left-hand/box rigid contacts; -1 outside pulse or on API failure."],["right_pair_count","count","pair audit","Matched right-hand/box rigid contacts; -1 outside pulse or on API failure."],["left_pair_normal_lambda_N","lambda","pair audit","Sum of RigidContact lambda for matched left-hand/box contacts."],["right_pair_normal_lambda_N","lambda","pair audit","Sum of RigidContact lambda for matched right-hand/box contacts."]];

const summaryDefs = [
["model","text","group key","Model label."],["direction","enum","group key","Perturbation direction."],["beta","ratio","group key","Requested beta."],["trials","count","sample size","All attempted trials in the cell, normally 5."],["precondition_success_rate","ratio","gate","Fraction of all trials that reached confirmed-carry gate."],["conditional_recovery_success_rate","ratio","outcome","Recovery rate only among precondition-success trials."],["termination_rate","ratio","outcome","Termination rate among precondition-success trials."],["drop_failure_rate","ratio","outcome","Drop/box-tilt rate among precondition-success trials."],["fall_failure_rate","ratio","outcome","Robot fall/base failure rate among precondition-success trials."],
["pulse_hold_retention_mean/std","ratio","aggregate","Sample mean/std of pulse confirmed-carry retention over valid trials."],["pulse_bimanual_contact_retention_mean/std","ratio","aggregate","Sample mean/std of pulse bimanual-contact retention."],["recovery_time_s_mean/std","s","aggregate","Mean/std over finite recovery times only; failures with NaN are excluded."],["post_confirmed_ratio_mean/std","ratio","aggregate","Mean/std post confirmed ratio; terminated trials contribute their stored truncated/zero value."],["goal_progress_m_mean/std","m","aggregate","Mean/std over finite values only; terminated trials are excluded, causing survivor bias."],["hand_box_rel_speed_rms_mps_mean/std","m/s","aggregate","Mean/std of trial-level relative-speed RMS."],["hand_box_rel_speed_peak_mps_mean/std","m/s","aggregate","Mean/std of trial-level relative-speed peak."],["max_abs_roll_rad_mean/std","rad","aggregate","Mean/std of absolute maximum roll over whole trial."],["max_abs_pitch_rad_mean/std","rad","aggregate","Mean/std of absolute maximum pitch over whole trial."],["left_hand_pulse_mean_N_mean/std","N","aggregate","Across-trial mean/std of each trial's pulse mean left proxy."],["right_hand_pulse_mean_N_mean/std","N","aggregate","Across-trial mean/std of pulse mean right proxy."],["resistive_hand_force_mean_N_mean/std","N","aggregate","Across-trial mean/std of signed resistive proxy."],["hand_load_asymmetry_mean_mean/std","ratio","aggregate","Across-trial mean/std of mean pulse load asymmetry."]];

const compareDefs = [
["positive_difference_means_interaction_minus_baseline","boolean","convention","All paired differences use interaction_priv - builtin."],["higher_is_better","list","interpretation","Metrics where a positive difference favors interaction_priv."],["lower_is_better","list","interpretation","Metrics where a negative difference favors interaction_priv."],["notes","text","limitation","Five seeds per cell are exploratory; no strong significance claim."],["cells","array","data","One record per direction x beta cell."],["direction","enum","cell key","Direction for this paired comparison."],["beta","ratio","cell key","Beta for this paired comparison."],["paired_seed_count","count","sample size","Seeds where both models passed the precondition gate."],["recovery_success_paired_mean_difference","ratio","paired delta","Mean paired interaction-minus-builtin recovery outcome."],["pulse_hold_retention_paired_mean_difference","ratio","paired delta","Mean paired delta in pulse hold retention."],["post_confirmed_ratio_paired_mean_difference","ratio","paired delta","Mean paired delta in post confirmed ratio."],["hand_box_rel_speed_rms_mps_paired_mean_difference","m/s","paired delta","Mean paired delta; negative favors interaction."],["max_abs_roll_rad_paired_mean_difference","rad","paired delta","Mean paired delta; negative favors interaction."],["max_abs_pitch_rad_paired_mean_difference","rad","paired delta","Mean paired delta; negative favors interaction."],["goal_progress_m_paired_mean_difference","m","paired delta","Mean paired delta; positive favors interaction, but terminated NaNs reduce pair count per metric."]];

function dictSheet(name, title, rows) {
  const s=wb.worksheets.add(name); s.showGridLines=false;
  s.getRange("A1:D2").merge(); s.getRange("A1").values=[[title]]; s.getRange("A1:D2").format=titleFmt;
  s.getRange("A4:D4").values=[["Variable","Unit/type","Stage","Exact meaning and caveat"]]; s.getRange("A4:D4").format=headFmt;
  s.getRangeByIndexes(4,0,rows.length,4).values=rows;
  s.getRange(`A5:A${rows.length+4}`).format.font={bold:true,color:navy};
  s.getRange(`A4:D${rows.length+4}`).format.wrapText=true;
  s.getRange("A:A").format.columnWidth=38; s.getRange("B:B").format.columnWidth=18; s.getRange("C:C").format.columnWidth=20; s.getRange("D:D").format.columnWidth=92;
  s.getRange(`A1:D${rows.length+4}`).format.autofitRows(); s.freezePanes.freezeRows(4);
}
dictSheet("Trials Dictionary","trials.csv — one row per independent attempted trial",trialDefs);
dictSheet("Trace Dictionary","force_trace.csv — one row per physics substep",traceDefs);
dictSheet("Summary Dictionary","summary.csv — model x direction x beta aggregation",summaryDefs);
dictSheet("Comparison Dictionary","comparison.json — paired interaction-minus-builtin differences",compareDefs);

const readme=wb.worksheets.add("README"); readme.showGridLines=false;
readme.getRange("A1:F2").merge(); readme.getRange("A1").values=[["How the perturbation experiment produces each record"]]; readme.getRange("A1:F2").format=titleFmt;
readme.getRange("A4:F12").values=[
["Step","Window","Duration","Operation","Primary file","Interpretation"],
[1,"Reset / precondition","up to 5 s","carryWith RSI; wait for 20 consecutive confirmed-carry policy steps","force_trace pre","No force is allowed before the gate"],
[2,"Schedule","instant","Freeze box-frame direction; Fpeak=min(beta*m*9.81, cap)","trials + trace","Direction is transformed once, not updated with box rotation"],
[3,"Pulse","0.10 s","20 midpoint-sampled physics substeps: F=Fpeak*sin(pi*(k+0.5)/20)","force_trace pulse","Force is applied at box rigid-body COM before gym.simulate; zero torque"],
[4,"Recovery","1.0 s","Require 5 consecutive confirmed-carry policy steps","trace + trials","Current recovery can succeed before a later drop"],
[5,"Continuation","2.0 s","Continue original carry task; task-success reset masked","trace + trials","Drop/fall/tilt resets remain active"],
[6,"Trial summary","one row","Compress trace into outcome, motion and contact-proxy metrics","trials.csv","NaN marks unavailable or excluded values"],
[7,"Cell summary","5 seeds","Group by model x direction x beta","summary.csv","Most metrics are conditional on precondition success"],
[8,"Paired comparison","3-5 paired seeds","interaction_priv minus builtin","comparison.json","Pair exists only when both models passed gate"]];
readme.getRange("A4:F4").format=headFmt; readme.getRange("A4:F12").format.wrapText=true;
readme.getRange("A14:F18").merge(); readme.getRange("A14").values=[[`Source files (not modified):\n${root}/trials.csv\n${root}/force_trace.csv\n${root}/summary.csv\n${root}/comparison.json`]]; readme.getRange("A14:F18").format={fill:gray,wrapText:true,font:{color:navy}};
for(const col of ["A","B","C","D","E","F"]) readme.getRange(`${col}:${col}`).format.columnWidth=col==="D"||col==="F"?44:22;
readme.getRange("A1:F18").format.autofitRows();

const inspect = await wb.inspect({kind:"table",range:"Dashboard!A1:G18",include:"values,formulas",tableMaxRows:18,tableMaxCols:7,maxChars:5000});
console.log(inspect.ndjson);
const errors = await wb.inspect({kind:"match",searchTerm:"#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",options:{useRegex:true,maxResults:100},summary:"formula error scan"});
console.log(errors.ndjson);
const preview = await wb.render({sheetName:"Dashboard",range:"A1:P61",scale:1.15,format:"png"});
await fs.writeFile(`${outDir}/boxperturb_ab_dashboard.png`,new Uint8Array(await preview.arrayBuffer()));
const forcePreview = await wb.render({sheetName:"Force and Hand Proxy",range:"A1:L45",scale:1.15,format:"png"});
await fs.writeFile(`${outDir}/boxperturb_force_hand.png`,new Uint8Array(await forcePreview.arrayBuffer()));
for (const [sheetName, range, fileName] of [
  ["README", "A1:F18", "qa_readme.png"],
  ["Trials Dictionary", "A1:D50", "qa_trials_dictionary.png"],
  ["Trace Dictionary", "A1:D45", "qa_trace_dictionary.png"],
  ["Summary Dictionary", "A1:D30", "qa_summary_dictionary.png"],
  ["Comparison Dictionary", "A1:D20", "qa_comparison_dictionary.png"],
]) {
  const rendered = await wb.render({sheetName, range, scale:0.85, format:"png"});
  await fs.writeFile(`${outDir}/${fileName}`,new Uint8Array(await rendered.arrayBuffer()));
}
const xlsx = await SpreadsheetFile.exportXlsx(wb);
await xlsx.save(`${outDir}/boxperturb_ab_audit.xlsx`);
console.log(`${outDir}/boxperturb_ab_audit.xlsx`);
