/* 自检脚本：核对 data.js / index.html 的资源引用与语法
 * 运行： node _check.js
 */
const fs = require('fs');
const path = require('path');

const DIR = __dirname;
const root = path.resolve(DIR, '..', '..', '..');   // 刘睿涵\刘睿涵
let bad = 0;
const ok = (c, msg) => { if (!c) bad++; console.log((c ? 'OK   ' : 'MISS ') + msg); };

/* ---- 1. data.js 语法 + 内容 ---- */
const src = fs.readFileSync(path.join(DIR, 'data.js'), 'utf8');
let NODES, EDGES, PDF_ROOT, NOTE_ROOT, BASIS;
try {
  ({ NODES, EDGES, PDF_ROOT, NOTE_ROOT, BASIS } = new Function(
    src + '\n;return {NODES, EDGES, PDF_ROOT, NOTE_ROOT, BASIS};')());
  console.log('OK   data.js 语法正确');
} catch (e) {
  console.log('FAIL data.js 语法错误: ' + e.message); process.exit(1);
}

console.log('\n--- 节点数: ' + NODES.length + '   边数: ' + EDGES.length + ' ---\n');

/* ---- 2. PDF 路径 ---- */
for (const n of NODES) {
  const p = path.resolve(DIR, PDF_ROOT, n.pdf);
  ok(fs.existsSync(p), `PDF  ${n.title.padEnd(16)} ${path.basename(p)}`);
}

/* ---- 3. 本地笔记路径 ---- */
for (const n of NODES) {
  for (const [lab, rel] of (n.notes || [])) {
    const p = path.resolve(DIR, NOTE_ROOT, rel);
    ok(fs.existsSync(p), `NOTE ${n.title.padEnd(16)} ${rel}`);
  }
}

/* ---- 4. 边引用完整性 + 依据合法性 ---- */
const ids = new Set(NODES.map(n => n.id));
for (const e of EDGES) {
  ok(ids.has(e.from) && ids.has(e.to) && BASIS[e.basis],
     `EDGE ${e.from} -> ${e.to}  [${e.basis}]`);
}

/* ---- 5. 飞书 token ---- */
for (const n of NODES) {
  const f = n.feishu || {};
  if (f.docx) console.log(`OK   FEISHU ${n.title.padEnd(16)} wiki=${f.wiki || '-'} docx=${f.docx}`);
}

/* ---- 6. index.html 内联脚本语法 ---- */
const html = fs.readFileSync(path.join(DIR, 'index.html'), 'utf8');
const blocks = [...html.matchAll(/<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/g)];
try {
  new Function(blocks.map(m => m[1]).join('\n'));
  console.log(`\nOK   index.html 内联脚本语法正确（${blocks.length} 段）`);
} catch (e) {
  console.log('\nFAIL index.html 内联脚本语法错误: ' + e.message); bad++;
}

/* ---- 7. 关键 DOM id 是否都存在 ---- */
for (const id of ['fbase', 'legend', 'dag', 'edges', 'panel']) {
  ok(html.includes(`id="${id}"`), `DOM  id="${id}"`);
}

console.log('\n' + (bad ? `❌ ${bad} 处问题` : '✅ 全部检查通过'));
process.exit(bad ? 1 : 0);
