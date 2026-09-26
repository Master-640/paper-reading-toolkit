/* 自检脚本：核对解析结果（papers/*.json）、关系声明、生成物与前端
 * 运行： node _check.js
 */
const fs = require('fs');
const path = require('path');

const DIR = __dirname;
let bad = 0;
const ok = (c, msg) => { if (!c) bad++; console.log((c ? 'OK   ' : 'FAIL ') + msg); };

/* ---------- 1. papers/*.json ---------- */
const pdir = path.join(DIR, 'papers');
const files = fs.readdirSync(pdir).filter(f => f.endsWith('.json'));
const papers = [];
for (const f of files) {
  try {
    const p = JSON.parse(fs.readFileSync(path.join(pdir, f), 'utf8'));
    papers.push(p);
    console.log(`OK   ${f} 解析成功  (id=${p.id})`);
  } catch (e) { ok(false, `${f} JSON 解析失败: ${e.message}`); }
}
const ids = new Set(papers.map(p => p.id));

const REQUIRED = ['id', 'title', 'venue', 'year', 'problem', 'methods'];
for (const p of papers) {
  const miss = REQUIRED.filter(k => !(k in p));
  ok(!miss.length, `${p.id} 必填字段${miss.length ? ' 缺: ' + miss.join(',') : '完整'}`);
  ok(!!p.problem?.task, `${p.id} problem.task 存在`);
  ok(!!p.problem?.domain?.length, `${p.id} problem.domain 非空`);
  ok(Array.isArray(p.methods) && p.methods.length > 0, `${p.id} methods 非空`);
}

/* ---------- 2. cites 指向仓库内、且带证据 ---------- */
let citeCount = 0;
for (const p of papers) {
  for (const c of (p.cites || [])) {
    citeCount++;
    ok(ids.has(c.target), `${p.id} cites → ${c.target} 目标存在`);
    ok(!!c.evidence && c.evidence.length > 20, `${p.id} cites → ${c.target} 附引用原文`);
  }
}
console.log(`\n--- 引用边（推导）: ${citeCount} 条 ---\n`);

/* ---------- 3. relations.json ---------- */
const rel = JSON.parse(fs.readFileSync(path.join(DIR, 'relations.json'), 'utf8'));
for (const e of rel.edges) {
  ok(ids.has(e.from) && ids.has(e.to), `声明边 ${e.from} → ${e.to} 两端存在`);
  ok(e.from !== e.to, `声明边 ${e.from} → ${e.to} 非自环`);
  ok(['same', 'method', 'infer'].includes(e.basis), `声明边 ${e.from} → ${e.to} basis 合法 (${e.basis})`);
  ok(!!e.reason, `声明边 ${e.from} → ${e.to} 附判定理由`);
}
console.log(`\n--- 声明边: ${rel.edges.length} 条 ---\n`);

/* ---------- 4. 生成物 data.js ---------- */
const df = path.join(DIR, 'data.js');
ok(fs.existsSync(df), 'data.js 存在（跑过 node build.js）');
if (fs.existsSync(df)) {
  let NODES, EDGES, BASIS;
  try {
    ({ NODES, EDGES, BASIS } = new Function(
      fs.readFileSync(df, 'utf8') + '\n;return {NODES, EDGES, BASIS};')());
    console.log(`OK   data.js 语法正确  (nodes=${NODES.length}, edges=${EDGES.length})`);
  } catch (e) { ok(false, 'data.js 语法错误: ' + e.message); }

  ok(NODES.length === papers.length, `data.js 节点数 == papers 数 (${NODES.length}/${papers.length})`);
  ok(EDGES.length === citeCount + rel.edges.length,
     `data.js 边数 == 引用(${citeCount}) + 声明(${rel.edges.length}) = ${citeCount + rel.edges.length}，实际 ${EDGES.length}`);

  for (const e of EDGES) {
    if (e.basis === 'explicit') ok(!!e.evidence, `边 ${e.from}→${e.to} 有引用原文`);
    else ok(!!e.reason, `边 ${e.from}→${e.to} 有判定理由`);
  }
}

/* ---------- 5. PDF 与笔记路径 ---------- */
const PDF_ROOT = '../../../papers/';
const NOTE_ROOT = '../';
for (const p of papers) {
  if (p.pdf) {
    const f = path.resolve(DIR, PDF_ROOT, p.pdf);
    ok(fs.existsSync(f), `PDF  ${p.id.padEnd(16)} ${p.pdf}`);
  }
  for (const [lab, r] of (p.notes || [])) {
    const f = path.resolve(DIR, NOTE_ROOT, r);
    ok(fs.existsSync(f), `NOTE ${p.id.padEnd(16)} ${r}`);
  }
}

/* ---------- 6. 前端 ---------- */
const html = fs.readFileSync(path.join(DIR, 'index.html'), 'utf8');
const blocks = [...html.matchAll(/<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/g)];
try {
  new Function(blocks.map(m => m[1]).join('\n'));
  console.log(`\nOK   index.html 内联脚本语法正确（${blocks.length} 段）`);
} catch (e) { ok(false, 'index.html 内联脚本语法错误: ' + e.message); }
for (const id of ['fbase', 'hideInfer', 'legend', 'dag', 'edges', 'panel']) {
  ok(html.includes(`id="${id}"`), `DOM  id="${id}"`);
}

console.log('\n' + (bad ? `❌ ${bad} 处问题` : '✅ 全部检查通过'));
process.exit(bad ? 1 : 0);
