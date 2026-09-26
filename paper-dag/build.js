#!/usr/bin/env node
/* ============================================================
 *  build.js —— 从解析结果构建 DAG
 *
 *  输入：
 *    papers/*.json       每篇论文按 schema.json 解析后的结构化结果
 *    relations.json      已声明（非引用）关系（basis=same/method/infer）
 *  输出：
 *    data.js             前端直接加载（NODES + EDGES + BASIS）
 *
 *  核心：**引用边是推导出来的，不是写死的。**
 *    - papers/*.json 的 cites 字段 → 生成 basis=explicit 的边，必须带 evidence
 *    - relations.json              → 生成 basis=same/method/infer 的边，必须带 reason
 *  另外输出「缺口报告」：字段高度重叠却没有边的论文对。
 *
 *  用法： node build.js
 * ============================================================ */
const fs = require('fs');
const path = require('path');

const DIR = __dirname;
const PAPERS_DIR = path.join(DIR, 'papers');
const OUT = path.join(DIR, 'data.js');

/* ---------- 读入 ---------- */
const papers = fs.readdirSync(PAPERS_DIR)
  .filter(f => f.endsWith('.json'))
  .map(f => {
    const p = path.join(PAPERS_DIR, f);
    try { return JSON.parse(fs.readFileSync(p, 'utf8')); }
    catch (e) { console.error(`✗ ${f} 解析失败: ${e.message}`); process.exit(1); }
  });

const declared = JSON.parse(fs.readFileSync(path.join(DIR, 'relations.json'), 'utf8')).edges;

const ids = new Set(papers.map(p => p.id));
const byId = Object.fromEntries(papers.map(p => [p.id, p]));
let errors = 0, warns = 0;
const err = m => { console.error('✗ ' + m); errors++; };
const warn = m => { console.warn('⚠ ' + m); warns++; };

/* ---------- 1. 校验必填字段 ---------- */
const REQUIRED = ['id', 'title', 'venue', 'year', 'problem', 'methods'];
for (const p of papers) {
  for (const k of REQUIRED) if (!(k in p)) err(`${p.id}: 缺必填字段 "${k}"`);
  if (!p.problem?.domain?.length) err(`${p.id}: problem.domain 为空`);
  if (!p.problem?.task) err(`${p.id}: problem.task 缺失`);
  if (!ids.has(p.id)) err(`${p.id}: id 重复`);
}

/* ---------- 2. 推导引用边 ---------- */
const edgeKey = (a, b) => a + '→' + b;
const edges = [];
const seen = new Map();

function addEdge(e, srcDesc) {
  const k = edgeKey(e.from, e.to);
  if (seen.has(k)) {
    const prev = seen.get(k);
    if (prev.basis === 'explicit' && e.basis !== 'explicit') {
      warn(`重复边 ${k}：保留引用边（explicit），忽略 ${srcDesc} 的声明`);
      return;
    }
    if (e.basis === 'explicit' && prev.basis === 'explicit') {
      warn(`重复引用边 ${k}`);
    } else {
      err(`重复边 ${k}：${prev._src} 与 ${srcDesc}`);
      return;
    }
  }
  e._src = srcDesc;
  seen.set(k, e);
  edges.push(e);
}

// 2a. cites → explicit（必须带 evidence）
for (const p of papers) {
  for (const c of (p.cites || [])) {
    if (!ids.has(c.target)) { err(`${p.id}.cites: target "${c.target}" 不在仓库内`); continue; }
    if (!c.evidence) { err(`${p.id}.cites → ${c.target}: 缺 evidence（引用边必须有可核对原文）`); continue; }
    addEdge({
      from: p.id, to: c.target, basis: 'explicit',
      relation: `引用（${c.relation || 'citation'}）`,
      evidence: c.evidence, where: c.where || '',
      source: 'derived:citation',
    }, `cites(${p.id})`);
  }
}

// 2b. declared → same / method / infer（必须带 reason）
for (const e of declared) {
  if (!ids.has(e.from)) { err(`relations.json: from "${e.from}" 不存在`); continue; }
  if (!ids.has(e.to)) { err(`relations.json: to "${e.to}" 不存在`); continue; }
  if (e.from === e.to) { err(`relations.json: 自环 ${e.from}`); continue; }
  if (e.basis === 'explicit') { err(`relations.json: ${e.from}→${e.to} 不该用 explicit（引用边应由 cites 推导）`); continue; }
  if (!e.reason) { err(`relations.json: ${e.from}→${e.to} 缺 reason`); continue; }
  if (!['same', 'method', 'infer'].includes(e.basis)) { err(`relations.json: ${e.from}→${e.to} basis 非法`); continue; }
  addEdge({
    from: e.from, to: e.to, basis: e.basis,
    relation: e.relation, reason: e.reason,
    bidirectional: !!e.bidirectional,
    source: 'declared',
  }, 'relations.json');
}

/* ---------- 3. 缺口报告 ---------- */
const connected = new Set();
for (const e of edges) { connected.add(edgeKey(e.from, e.to)); connected.add(edgeKey(e.to, e.from)); }
const gaps = [];
for (let i = 0; i < papers.length; i++) {
  for (let j = i + 1; j < papers.length; j++) {
    const A = papers[i], B = papers[j];
    if (connected.has(edgeKey(A.id, B.id))) continue;
    const sharedDomain = (A.problem.domain || []).filter(d => (B.problem.domain || []).includes(d));
    const sharedMethod = (A.methods || []).filter(m => (B.methods || []).includes(m));
    const sameTask = A.problem.task === B.problem.task;
    if (sharedDomain.length || sharedMethod.length || sameTask) {
      gaps.push({ pair: `${A.id} ↔ ${B.id}`, sharedDomain, sharedMethod, sameTask });
    }
  }
}
// 另有：字段重叠很低却声明了边的
const weak = edges.filter(e => {
  if (e.basis === 'explicit') return false;
  const A = byId[e.from], B = byId[e.to];
  const sd = (A.problem.domain || []).filter(d => (B.problem.domain || []).includes(d));
  const sm = (A.methods || []).filter(m => (B.methods || []).includes(m));
  return sd.length === 0 && sm.length === 0;
});

/* ---------- 4. 组装 NODES ---------- */
const NODES = papers.map(p => ({
  id: p.id,
  title: p.short || p.title,
  fullTitle: p.title,
  subtitle: `${p.authors ? p.authors.split(',')[0] + ' et al. · ' : ''}${p.venue} · ${p.year}`,
  col: p.pos?.col ?? 0,
  row: p.pos?.row ?? 0,
  kind: p.role || '',
  accent: p.accent || '#2563eb',
  oneline: p.oneline || '',
  type: p.type || '',
  cid: p.doi || '',
  pdf: p.pdf || '',
  feishu: p.feishu || {},
  problem: p.problem || {},
  methods: p.methods || [],
  data: p.data || {},
  contributions: p.contributions || [],
  numbers: p.numbers || [],
  baselines: p.baselines || [],
  limitations: p.limitations || [],
  reusable: p.reusable || [],
  notes: p.notes || [],
  cites: p.cites || [],
}));

/* ---------- 5. 写 data.js ---------- */
const banner = `/* 自动生成，请勿手改 —— 源文件在 papers/*.json 与 relations.json
 * 重新生成： node build.js
 * 生成时间： ${new Date().toISOString().replace('T', ' ').slice(0, 19)}
 */`;

const out = `${banner}

const FEISHU_BASE_DEFAULT = "https://feishu.cn";
const PDF_ROOT = "../../../papers/";
const NOTE_ROOT = "../";

const BASIS = {
  explicit: { label: "明确", color: "#dc2626", dash: "", desc: "论文显式引用（附可核对原文）" },
  same:     { label: "同题", color: "#2563eb", dash: "", desc: "解决同一问题，方法不同" },
  method:   { label: "方法", color: "#16a34a", dash: "8 6", desc: "方法/技术亲缘" },
  infer:    { label: "推断", color: "#94a3b8", dash: "2 5", desc: "阅读地图推断，无文献证据" },
};

const NODES = ${JSON.stringify(NODES, null, 2)};

const EDGES = ${JSON.stringify(edges.map(({ _src, ...e }) => e), null, 2)};
`;
fs.writeFileSync(OUT, out, 'utf8');

/* ---------- 报告 ---------- */
console.log('='.repeat(66));
console.log(`  papers : ${papers.length}    edges : ${edges.length}`);
const cnt = {};
for (const e of edges) cnt[e.basis] = (cnt[e.basis] || 0) + 1;
console.log('  basis  : ' + Object.entries(cnt).map(([k, v]) => `${k}=${v}`).join('  '));
console.log(`  derived(citation)=${edges.filter(e => e.source === 'derived:citation').length}  declared=${edges.filter(e => e.source === 'declared').length}`);
console.log('='.repeat(66));
for (const e of edges) {
  const tag = e.basis === 'explicit' ? '引用' : '声明';
  console.log(`  [${tag}] ${e.from} → ${e.to}  (${e.basis})`);
  console.log(`         ${e.relation}`);
}
if (gaps.length) {
  console.log('\n--- 缺口报告：字段重叠但无边 ---');
  for (const g of gaps) {
    console.log(`  ${g.pair}   domain=[${g.sharedDomain}]  method=[${g.sharedMethod}]  sameTask=${g.sameTask}`);
  }
}
if (weak.length) {
  console.log('\n--- 弱边提示：已声明边但字段无重叠（靠 reason 支撑）---');
  for (const e of weak) console.log(`  ${e.from} → ${e.to}  (${e.basis})`);
}
console.log(errors ? `\n❌ ${errors} error(s), ${warns} warning(s)` : `\n✅ 构建通过，${warns} warning(s)`);
console.log(`   → ${path.relative(process.cwd(), OUT)}`);
process.exit(errors ? 1 : 0);
