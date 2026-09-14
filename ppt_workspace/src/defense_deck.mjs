/** @jsxRuntime automatic */
/** @jsxImportSource @oai/artifact-tool/presentation-jsx */

import path from "node:path";
import fs from "node:fs";
import {
  Presentation,
  PresentationFile,
  column,
  row,
  grid,
  panel,
  text,
  image,
  rule,
  fill,
  hug,
  fixed,
  wrap,
  grow,
  fr,
} from "@oai/artifact-tool";

const ROOT = path.resolve("..");
const OUT = path.resolve("output", "脑肿瘤MRI精细分割算法研究_答辩PPT.pptx");
const VIS = path.resolve(ROOT, "result_500", "summary_0.5", "results_final", "visualize");

const C = {
  ink: "#16211D",
  muted: "#5C6B62",
  bg: "#F7F4EC",
  paper: "#FFFFFF",
  green: "#1F8A5B",
  yellow: "#E6B325",
  red: "#C84630",
  blue: "#28536B",
  pale: "#E7EFE7",
  line: "#D5D0C4",
};

const W = 1920;
const H = 1080;

const presentation = Presentation.create({
  slideSize: { width: W, height: H },
});

function pngDataUrl(filePath) {
  return `data:image/png;base64,${fs.readFileSync(filePath).toString("base64")}`;
}

function slideRoot(slide, children, fillColor = C.bg) {
  slide.compose(
    panel(
      { name: "root-bg", width: fill, height: fill, fill: fillColor },
      column(
        {
          name: "root",
          width: fill,
          height: fill,
          padding: { x: 88, y: 62 },
          gap: 28,
        },
        children,
      ),
    ),
    { frame: { left: 0, top: 0, width: W, height: H }, baseUnit: 8 },
  );
}

function titleBlock(title, subtitle = "") {
  const nodes = [
    text(title, {
      name: "slide-title",
      width: fill,
      height: hug,
      style: { fontSize: 55, bold: true, color: C.ink },
    }),
    rule({ name: "title-rule", width: fixed(190), stroke: C.green, weight: 5 }),
  ];
  if (subtitle) {
    nodes.push(
      text(subtitle, {
        name: "slide-subtitle",
        width: wrap(1400),
        height: hug,
        style: { fontSize: 25, color: C.muted },
      }),
    );
  }
  return column({ name: "title-block", width: fill, height: hug, gap: 14 }, nodes);
}

function bullets(items, size = 31) {
  return text(items.map((item) => `• ${item}`).join("\n"), {
    name: "bullets",
    width: fill,
    height: hug,
    style: { fontSize: size, color: C.ink },
  });
}

function bigNumber(value, label, color = C.green) {
  return column(
    { name: `metric-${label}`, width: fill, height: hug, gap: 6 },
    [
      text(value, {
        name: "metric-value",
        width: fill,
        height: hug,
        style: { fontSize: 62, bold: true, color },
      }),
      text(label, {
        name: "metric-label",
        width: fill,
        height: hug,
        style: { fontSize: 22, color: C.muted },
      }),
    ],
  );
}

function card(child, name = "card") {
  return panel(
    {
      name,
      width: fill,
      height: fill,
      padding: { x: 30, y: 26 },
      fill: C.paper,
      borderRadius: "rounded-lg",
    },
    child,
  );
}

function twoColumnSlide(title, left, right, subtitle = "") {
  const slide = presentation.slides.add();
  slideRoot(slide, [
    titleBlock(title, subtitle),
    grid(
      {
        name: "two-cols",
        width: fill,
        height: fill,
        columns: [fr(1), fr(1)],
        rows: [fr(1)],
        columnGap: 44,
      },
      [left, right],
    ),
  ]);
}

function tableText(headers, rows, fontSize = 24) {
  const lines = [
    headers.join("  |  "),
    headers.map(() => "---").join("  |  "),
    ...rows.map((r) => r.join("  |  ")),
  ];
  return text(lines.join("\n"), {
    name: "table-text",
    width: fill,
    height: hug,
    style: { fontSize, color: C.ink },
  });
}

function addCover() {
  const slide = presentation.slides.add();
  slideRoot(
    slide,
    [
      row(
        { name: "cover-row", width: fill, height: fill, gap: 40 },
        [
          column(
            { name: "cover-copy", width: grow(1.05), height: fill, justify: "center", gap: 26 },
            [
              text("脑肿瘤 MRI\n精细分割算法研究", {
                name: "cover-title",
                width: fill,
                height: hug,
                style: { fontSize: 78, bold: true, color: C.ink },
              }),
              rule({ name: "cover-rule", width: fixed(260), stroke: C.green, weight: 7 }),
              text("基于 U-Net 的 2D / 2.5D 分割、Boundary Loss 与 3D 后处理评估", {
                name: "cover-subtitle",
                width: wrap(920),
                height: hug,
                style: { fontSize: 30, color: C.muted },
              }),
              text("严宏伟 · 毕业设计答辩", {
                name: "cover-author",
                width: fill,
                height: hug,
                style: { fontSize: 25, color: C.blue },
              }),
            ],
          ),
          card(
            column(
              { name: "cover-metrics", width: fill, height: fill, justify: "center", gap: 34 },
              [
                bigNumber("500", "patients used in formal experiment", C.green),
                bigNumber("3", "random seeds, patient-level split", C.blue),
                bigNumber("4", "model variants compared", C.red),
              ],
            ),
            "cover-metric-card",
          ),
        ],
      ),
    ],
    "#EFE8D8",
  );
}

addCover();

twoColumnSlide(
  "研究背景与问题",
  column({ width: fill, height: fill, gap: 24 }, [
    bullets([
      "脑肿瘤 MRI 分割是诊断、治疗计划和疗效评估的重要基础。",
      "临床上关心的不只是整体 Dice，也关心漏检、误检和边界定位。",
      "BraTS 数据中肿瘤形态差异大，小体积病灶更容易受切片噪声影响。",
    ]),
  ]),
  card(
    column({ width: fill, height: fill, justify: "center", gap: 28 }, [
      bigNumber("240 × 240 × 155", "single patient 3D volume shape", C.blue),
      bigNumber("2,808–361,783", "tumor voxel range in scanned dataset", C.green),
    ]),
    "background-stats",
  ),
  "核心矛盾：既要提高肿瘤检出，又要控制误报结构对 3D 指标的破坏。",
);

twoColumnSlide(
  "研究目标",
  card(
    column({ width: fill, height: fill, gap: 24 }, [
      text("本课题回答三个问题", {
        width: fill,
        height: hug,
        style: { fontSize: 36, bold: true, color: C.ink },
      }),
      bullets([
        "2.5D 输入能否比单切片 2D 更稳定？",
        "Boundary Loss 是否能改善边界和漏检？",
        "3D 连通域后处理能否降低 HD95 的极端误差？",
      ]),
    ]),
    "goal-card",
  ),
  column({ width: fill, height: fill, justify: "center", gap: 18 }, [
    text("评价口径", {
      width: fill,
      height: hug,
      style: { fontSize: 38, bold: true, color: C.green },
    }),
    bullets([
      "patient-level 3D Dice",
      "true 3D HD95",
      "slice-level FNR / FPR",
      "Wilcoxon paired validation",
    ]),
  ]),
);

twoColumnSlide(
  "方法总览",
  column({ width: fill, height: fill, gap: 22 }, [
    text("U-Net 主干", { width: fill, height: hug, style: { fontSize: 38, bold: true, color: C.ink } }),
    bullets([
      "2D：输入单张 FLAIR 切片。",
      "2.5D：输入相邻切片堆叠，引入上下文但保持 2D 计算成本。",
      "输出二值肿瘤区域，阈值固定为 0.5。",
    ]),
  ]),
  card(
    column({ width: fill, height: fill, justify: "center", gap: 22 }, [
      text("Loss 设计", { width: fill, height: hug, style: { fontSize: 38, bold: true, color: C.ink } }),
      bullets([
        "BCE：像素级分类监督。",
        "Dice Loss：缓解前景/背景不平衡。",
        "Boundary Loss：强化边界区域误差。",
      ]),
    ]),
    "loss-card",
  ),
);

twoColumnSlide(
  "实验设计",
  card(
    column({ width: fill, height: fill, gap: 22 }, [
      text("正式实验设置", { width: fill, height: hug, style: { fontSize: 36, bold: true, color: C.ink } }),
      tableText(
        ["项目", "设置"],
        [
          ["患者数", "500"],
          ["seed", "0 / 1 / 2"],
          ["划分", "train : val : test = 7 : 1 : 2"],
          ["test", "每个 seed 100 patients"],
          ["模型数", "4 组"],
        ],
        27,
      ),
    ]),
    "exp-card",
  ),
  column({ width: fill, height: fill, gap: 18 }, [
    text("四组模型", { width: fill, height: hug, style: { fontSize: 36, bold: true, color: C.green } }),
    bullets([
      "2D Dice+BCE",
      "2D Dice+BCE+Boundary",
      "2.5D Dice+BCE",
      "2.5D Dice+BCE+Boundary",
    ], 32),
  ]),
);

twoColumnSlide(
  "原始 3D 结果",
  card(
    tableText(
      ["模型", "Dice mean", "HD95 median"],
      [
        ["2D BCE+Dice", "0.870", "6.63"],
        ["2D + Boundary", "0.870", "14.4"],
        ["2.5D BCE+Dice", "0.890", "4.12"],
        ["2.5D + Boundary", "0.881", "7.93"],
      ],
      28,
    ),
    "original-table",
  ),
  column({ width: fill, height: fill, justify: "center", gap: 22 }, [
    text("主要观察", { width: fill, height: hug, style: { fontSize: 38, bold: true, color: C.ink } }),
    bullets([
      "2.5D Dice+BCE 在 Dice 和 HD95 上整体优于 2D。",
      "Boundary Loss 降低漏检，但原始预测中会引入更多误检结构。",
      "HD95 对远离 GT 的小 FP 结构非常敏感。",
    ]),
  ]),
);

twoColumnSlide(
  "漏检与误检：Boundary 的双刃剑",
  card(
    tableText(
      ["模型", "FNR mean", "FPR mean"],
      [
        ["2D BCE+Dice", "0.0569", "0.159"],
        ["2D + Boundary", "0.0176", "0.387"],
        ["2.5D BCE+Dice", "0.0820", "0.0398"],
        ["2.5D + Boundary", "0.0181", "0.314"],
      ],
      28,
    ),
    "fnr-fpr-table",
  ),
  column({ width: fill, height: fill, justify: "center", gap: 22 }, [
    text("答辩时的关键解释", {
      width: fill,
      height: hug,
      style: { fontSize: 38, bold: true, color: C.red },
    }),
    bullets([
      "Boundary 让模型更愿意预测边界附近前景，因此 FNR 明显下降。",
      "代价是空切片上更容易出现小面积假阳性，导致 FPR 上升。",
      "这也是后处理被纳入最终分析的原因。",
    ]),
  ]),
);

const chartDice = path.join(VIS, "dice_mean_original_vs_cc500_vs_top30.png");
const chartHd95 = path.join(VIS, "hd95_median_original_vs_cc500_vs_top30.png");

twoColumnSlide(
  "3D 连通域后处理",
  image({
    name: "dice-chart",
    dataUrl: pngDataUrl(chartDice),
    contentType: "image/png",
    width: fill,
    height: fill,
    fit: "contain",
    alt: "Dice mean comparison after postprocessing",
  }),
  image({
    name: "hd95-chart",
    dataUrl: pngDataUrl(chartHd95),
    contentType: "image/png",
    width: fill,
    height: fill,
    fit: "contain",
    alt: "HD95 median comparison after postprocessing",
  }),
  "删除小连通域后，HD95 的极端误差被明显压低；Dice 整体保持或小幅改善。",
);

twoColumnSlide(
  "cc500 后处理后的 pooled 结果",
  card(
    tableText(
      ["模型", "Dice mean", "HD95 median"],
      [
        ["2D BCE+Dice", "0.877", "4.47"],
        ["2D + Boundary", "0.881", "5.10"],
        ["2.5D BCE+Dice", "0.892", "4.12"],
        ["2.5D + Boundary", "0.889", "4.12"],
      ],
      28,
    ),
    "cc500-table",
  ),
  column({ width: fill, height: fill, justify: "center", gap: 22 }, [
    text("后处理的意义", { width: fill, height: hug, style: { fontSize: 38, bold: true, color: C.green } }),
    bullets([
      "小 FP 连通域会显著拉高 HD95。",
      "cc500 清理的是预测体中的离散噪点，不改变 GT。",
      "处理后 2.5D 仍保持最优或并列最优的 HD95 表现。",
    ]),
  ]),
);

twoColumnSlide(
  "按肿瘤体积分组：小病灶最难",
  card(
    tableText(
      ["分组", "2D Dice", "2.5D Dice", "2.5D+Boundary Dice"],
      [
        ["small 30%", "0.818", "0.846", "0.837"],
        ["mid 40%", "0.893", "0.907", "0.906"],
        ["large 30%", "0.915", "0.918", "0.919"],
      ],
      25,
    ),
    "size-table",
  ),
  column({ width: fill, height: fill, justify: "center", gap: 22 }, [
    text("结论指向", { width: fill, height: hug, style: { fontSize: 38, bold: true, color: C.ink } }),
    bullets([
      "体积越小，Dice 越低，HD95 越不稳定。",
      "2.5D 对小病灶更有帮助，说明上下文信息确实有价值。",
      "Boundary 对小病灶不总是稳定，需配合后处理和阈值策略。",
    ]),
  ]),
);

const caseLarge = path.join(VIS, "seed0_BraTS2021_00147.png");
const caseMid = path.join(VIS, "seed0_BraTS2021_00071.png");

twoColumnSlide(
  "三维可视化示例",
  image({
    name: "case-large",
    dataUrl: pngDataUrl(caseLarge),
    contentType: "image/png",
    width: fill,
    height: fill,
    fit: "contain",
    alt: "3D visualization case BraTS2021_00147",
  }),
  image({
    name: "case-mid",
    dataUrl: pngDataUrl(caseMid),
    contentType: "image/png",
    width: fill,
    height: fill,
    fit: "contain",
    alt: "3D visualization case BraTS2021_00071",
  }),
  "绿色为 GT，黄色为模型预测；左侧 GT，右侧按 2D / 2.5D 模型排列。",
);

twoColumnSlide(
  "统计检验",
  card(
    tableText(
      ["比较", "指标", "p-value"],
      [
        ["2.5D+Boundary vs 2D+Boundary", "Dice", "2.87e-08"],
        ["2.5D+Boundary vs 2D+Boundary", "HD95", "9.89e-04"],
        ["2.5D+Boundary vs 2.5D BCE+Dice", "Dice", "0.210"],
        ["2.5D+Boundary vs 2.5D BCE+Dice", "HD95", "3.20e-04"],
      ],
      24,
    ),
    "wilcoxon-table",
  ),
  column({ width: fill, height: fill, justify: "center", gap: 22 }, [
    text("验证方式", { width: fill, height: hug, style: { fontSize: 38, bold: true, color: C.blue } }),
    bullets([
      "同一批 test patient 上做配对检验。",
      "三组 seed 合并为 300 个 patient-level paired samples。",
      "使用 Wilcoxon signed-rank test，避免正态性假设过强。",
    ]),
  ]),
);

twoColumnSlide(
  "创新点与贡献",
  column({ width: fill, height: fill, gap: 24 }, [
    bullets([
      "构建了 2D / 2.5D U-Net 在 BraTS2021 FLAIR 上的可复现实验流程。",
      "把 Dice、HD95、FNR、FPR 结合起来评价模型，而不是只看单一 Dice。",
      "分析了 Boundary Loss 对漏检和误检的双向影响。",
      "加入 patient-level 3D 后处理和体积分组分析，解释 HD95 变化来源。",
    ]),
  ]),
  card(
    column({ width: fill, height: fill, justify: "center", gap: 26 }, [
      text("最重要的结论", {
        width: fill,
        height: hug,
        style: { fontSize: 44, bold: true, color: C.green },
      }),
      text("2.5D 输入提供了更稳定的空间上下文；Boundary Loss 能降低漏检，但必须配合后处理控制误报。", {
        width: fill,
        height: hug,
        style: { fontSize: 34, color: C.ink },
      }),
    ]),
    "contribution-summary",
  ),
);

twoColumnSlide(
  "不足与展望",
  card(
    column({ width: fill, height: fill, gap: 22 }, [
      text("当前不足", { width: fill, height: hug, style: { fontSize: 38, bold: true, color: C.red } }),
      bullets([
        "只使用 FLAIR 单模态，没有融合 T1/T1ce/T2。",
        "2.5D 仍不是完整 3D 网络，跨层空间建模有限。",
        "Boundary 权重和后处理阈值仍需更系统的外部验证。",
      ]),
    ]),
    "limits-card",
  ),
  column({ width: fill, height: fill, justify: "center", gap: 22 }, [
    text("后续方向", { width: fill, height: hug, style: { fontSize: 38, bold: true, color: C.green } }),
    bullets([
      "多模态输入与 3D/nnU-Net 基线比较。",
      "面向小病灶的采样与损失函数改进。",
      "把连通域后处理改为可学习或验证集自适应策略。",
    ]),
  ]),
);

const slide = presentation.slides.add();
slideRoot(
  slide,
  [
    column(
      { name: "ending", width: fill, height: fill, justify: "center", gap: 34 },
      [
        text("谢谢各位老师", {
          name: "thanks",
          width: fill,
          height: hug,
          style: { fontSize: 78, bold: true, color: C.ink },
        }),
        rule({ name: "thanks-rule", width: fixed(220), stroke: C.green, weight: 7 }),
        text("欢迎批评指正", {
          name: "qa",
          width: fill,
          height: hug,
          style: { fontSize: 38, color: C.muted },
        }),
      ],
    ),
  ],
  "#EFE8D8",
);

const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(OUT);
console.log(OUT);
