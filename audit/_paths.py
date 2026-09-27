"""审计脚本的路径解析。

论文 PDF 不在本仓库内（版权原因，见根目录 .gitignore），
SI 放在本仓库根目录。可用环境变量覆盖：

    UNICM_PDF       主文 PDF（默认在仓库外的 ../papers/ 下）
    UNICM_SI_PDF    补充材料 PDF（MOESM1，在仓库根目录）
    UNICM_AUDIT_OUT 输出目录（默认 audit/out/，已被 .gitignore 忽略）

目录结构假设（本目录位于 <仓库根>/audit/）：

    <外层>/papers/UniCM.pdf
    <外层>/<仓库名>/UniCM_src_SI_MOESM1.pdf
    <外层>/<仓库名>/audit/            <- 本目录
"""
import os
import pathlib

_AUDIT = pathlib.Path(__file__).resolve().parent          # <仓库根>/audit
_REPO = _AUDIT.parent                                     # <仓库根>
_OUTER = _REPO.parent.parent                              # 外层（papers 的父目录）

MAIN_PDF = os.environ.get("UNICM_PDF", str(_OUTER / "papers" / "UniCM.pdf"))
SI_PDF = os.environ.get("UNICM_SI_PDF", str(_REPO / "UniCM_src_SI_MOESM1.pdf"))

OUT = pathlib.Path(os.environ.get("UNICM_AUDIT_OUT", str(_AUDIT / "out")))
SI_FIGS = str(OUT / "si_figs")
FIG3 = str(OUT / "fig3")

for _d in (OUT, pathlib.Path(SI_FIGS), pathlib.Path(FIG3)):
    _d.mkdir(parents=True, exist_ok=True)


def require_main_pdf() -> str:
    """找不到主文 PDF 时给出可操作的提示，而不是抛一个裸的 FileNotFoundError。"""
    if not pathlib.Path(MAIN_PDF).exists():
        raise SystemExit(
            f"找不到主文 PDF：{MAIN_PDF}\n"
            f"请用 UNICM_PDF=<路径> 指定，或放到 {_OUTER / 'papers'} 下。"
        )
    return MAIN_PDF


def require_si_pdf() -> str:
    """找不到 SI 时同理。"""
    if not pathlib.Path(SI_PDF).exists():
        raise SystemExit(
            f"找不到补充材料 PDF：{SI_PDF}\n"
            f"请用 UNICM_SI_PDF=<路径> 指定。"
        )
    return SI_PDF
