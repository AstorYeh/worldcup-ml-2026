// 2026 世界盃 ML 預測系統 — 期末專題報告 PPT
// 設計風格：Coral Energy (足球能量) + 暗色封面 + 淺色內容（sandwich）
const pptxgen = require("pptxgenjs");
const pres = new pptxgen();

pres.layout = "LAYOUT_WIDE"; // 13.3" × 7.5"
pres.author = "Jay Yeh";
pres.title = "2026 世界盃 ML 預測系統 — 期末專題報告";

// ── 配色 ─────────────────────────────────────────────
const C = {
  primary: "F96167",   // 珊瑚紅（主色）
  navy:    "1A2B5C",   // 深海軍藍（背景/標題）
  gold:    "F9C846",   // 金色（強調）
  white:   "FFFFFF",
  cream:   "FFF8F3",   // 暖白（內容背景）
  ink:     "1A1A1A",   // 主文字
  muted:   "6B6760",   // 副文字
  border:  "E8E5DD",   // 邊框淺灰
  green:   "5F8466",   // 正向/成功
  blue:    "1F6E8C",   // 數據藍
};

// ── 字型 ─────────────────────────────────────────────
const FONT_TITLE = "Calibri";
const FONT_SERIF = "Georgia";
const FONT_BODY  = "Calibri";

const SLIDE_W = 13.3;
const SLIDE_H = 7.5;

// ── 共用 helper ──────────────────────────────────────
function addFooter(slide, pageNum, total) {
  // 底部頁碼
  slide.addText(`${pageNum} / ${total}`, {
    x: 12.3, y: 7.15, w: 0.8, h: 0.3,
    fontSize: 9, color: C.muted, align: "right", fontFace: FONT_BODY,
  });
  slide.addText("2026 World Cup ML Prediction System", {
    x: 0.5, y: 7.15, w: 6, h: 0.3,
    fontSize: 9, color: C.muted, fontFace: FONT_BODY,
  });
  // 底部裝飾線
  slide.addShape(pres.shapes.LINE, {
    x: 0.5, y: 7.1, w: 12.3, h: 0,
    line: { color: C.border, width: 0.5 },
  });
}

function addTitle(slide, title, subtitle) {
  // 左邊珊瑚紅短條
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 0.55, w: 0.12, h: 0.55,
    fill: { color: C.primary }, line: { type: "none" },
  });
  slide.addText(title, {
    x: 0.78, y: 0.45, w: 11, h: 0.7,
    fontSize: 28, bold: true, color: C.navy,
    fontFace: FONT_TITLE, margin: 0,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.78, y: 1.08, w: 11, h: 0.4,
      fontSize: 13, color: C.muted,
      fontFace: FONT_BODY, italic: true, margin: 0,
    });
  }
  // 分隔線
  slide.addShape(pres.shapes.LINE, {
    x: 0.5, y: 1.55, w: 12.3, h: 0,
    line: { color: C.border, width: 0.5 },
  });
}

function addSectionLabel(slide, text) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 0.3, w: 1.8, h: 0.3,
    fill: { color: C.primary }, line: { type: "none" },
  });
  slide.addText(text, {
    x: 0.5, y: 0.3, w: 1.8, h: 0.3,
    fontSize: 10, bold: true, color: C.white, align: "center",
    fontFace: FONT_BODY, charSpacing: 2,
  });
}

const TOTAL = 24;

// ═════════════════════════════════════════════════════════════════
// SLIDE 1: 封面（深色設計）
// ═════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.navy };

  // 左上 PART 標籤
  s.addText("FINAL PROJECT · 2026", {
    x: 0.5, y: 0.5, w: 6, h: 0.4,
    fontSize: 11, color: C.gold, charSpacing: 4, bold: true, fontFace: FONT_BODY,
  });

  // 主標題
  s.addText("2026 世界盃", {
    x: 0.5, y: 2.0, w: 12, h: 1.4,
    fontSize: 72, bold: true, color: C.white, fontFace: FONT_TITLE, margin: 0,
  });
  s.addText("ML 勝率分析與比分預測系統", {
    x: 0.5, y: 3.3, w: 12, h: 0.9,
    fontSize: 36, color: C.primary, fontFace: FONT_TITLE, bold: true, margin: 0,
  });

  // 副標：技術 stack
  s.addText("XGBoost · Dixon-Coles Poisson · Monte Carlo · K-Means Clustering", {
    x: 0.5, y: 4.4, w: 12, h: 0.4,
    fontSize: 16, color: C.cream, fontFace: FONT_SERIF, italic: true, margin: 0,
  });

  // 裝飾線
  s.addShape(pres.shapes.LINE, {
    x: 0.5, y: 5.2, w: 3.5, h: 0,
    line: { color: C.gold, width: 3 },
  });

  // 數據亮點
  s.addText([
    { text: "49,328", options: { bold: true, color: C.gold, fontSize: 28, fontFace: FONT_TITLE } },
    { text: "  場國際比賽\n", options: { color: C.cream, fontSize: 12, breakLine: true } },
    { text: "67,894", options: { bold: true, color: C.gold, fontSize: 28, fontFace: FONT_TITLE } },
    { text: "  筆 FIFA 排名\n", options: { color: C.cream, fontSize: 12, breakLine: true } },
    { text: "240", options: { bold: true, color: C.gold, fontSize: 28, fontFace: FONT_TITLE } },
    { text: "  位主將陣容資料", options: { color: C.cream, fontSize: 12 } },
  ], {
    x: 0.5, y: 5.4, w: 7, h: 1.5, fontFace: FONT_BODY, margin: 0,
  });

  // 右下 作者區
  s.addText("Jay Yeh", {
    x: 8.5, y: 6.3, w: 4.3, h: 0.4,
    fontSize: 18, bold: true, color: C.white, align: "right", fontFace: FONT_TITLE,
  });
  s.addText("May 2026 · 期末專題報告", {
    x: 8.5, y: 6.7, w: 4.3, h: 0.3,
    fontSize: 11, color: C.cream, align: "right", fontFace: FONT_BODY,
  });
}

// ═════════════════════════════════════════════════════════════════
// SLIDE 2: 目錄
// ═════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.cream };
  addTitle(s, "目錄 · Agenda", "8 個章節，從資料工程到模型驗證的完整流程");

  const items = [
    ["01", "專題背景與動機",    "為什麼預測世界盃？挑戰在哪？"],
    ["02", "資料工程",          "49,328 場 + 67,894 排名 + 240 球員"],
    ["03", "四層融合模型架構",  "XGBoost + Dixon-Coles + λ修正 + MAP"],
    ["04", "模型驗證",          "Walk-Forward + 混淆矩陣 + ROC + 費雪檢定"],
    ["05", "球隊風格分群",      "K-Means + PCA + 攻/守/平衡 三類型"],
    ["06", "奪冠機率預測",      "Monte Carlo 10,000 次完整賽程模擬"],
    ["07", "技術創新與挑戰",    "公式 bug 排除、平局悖論、數據校正"],
    ["08", "結論與未來工作",    "主要貢獻、系統限制、改進方向"],
  ];

  // 兩欄式排版
  const colWidth = 5.8;
  const rowH = 0.65;
  items.forEach((item, i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const xBase = 0.6 + col * (colWidth + 0.3);
    const yBase = 2.0 + row * (rowH + 0.25);
    // 數字徽章
    s.addShape(pres.shapes.OVAL, {
      x: xBase, y: yBase, w: 0.55, h: 0.55,
      fill: { color: C.primary }, line: { type: "none" },
    });
    s.addText(item[0], {
      x: xBase, y: yBase, w: 0.55, h: 0.55,
      fontSize: 14, bold: true, color: C.white, align: "center", valign: "middle",
      fontFace: FONT_TITLE, margin: 0,
    });
    // 標題
    s.addText(item[1], {
      x: xBase + 0.75, y: yBase + 0.02, w: colWidth - 0.8, h: 0.32,
      fontSize: 15, bold: true, color: C.navy, fontFace: FONT_TITLE, margin: 0,
    });
    // 副標
    s.addText(item[2], {
      x: xBase + 0.75, y: yBase + 0.32, w: colWidth - 0.8, h: 0.28,
      fontSize: 11, color: C.muted, fontFace: FONT_BODY, margin: 0,
    });
  });

  addFooter(s, 2, TOTAL);
}

// ═════════════════════════════════════════════════════════════════
// PART 1: 專題背景與動機
// ═════════════════════════════════════════════════════════════════

// SLIDE 3: 為什麼選擇世界盃預測
{
  const s = pres.addSlide();
  s.background = { color: C.cream };
  addSectionLabel(s, "PART 01");
  addTitle(s, "為什麼選擇 2026 世界盃？", "全球運動賽事預測是機器學習的經典難題");

  // 三個亮點（橫向 3 卡片）
  const cards = [
    { icon: "🌍", title: "全球規模盛事", desc: "48 隊 · 104 場比賽\n累計超過 30 億觀眾收看" },
    { icon: "🎯", title: "預測難度高", desc: "平局率 ~25% · 黑馬常見\n商業博弈業界共同難題" },
    { icon: "📊", title: "資料豐富",     desc: "完整歷史比賽資料\nFIFA 排名 + 球員能力公開可得" },
  ];
  const cw = 3.95;
  cards.forEach((c, i) => {
    const x = 0.6 + i * (cw + 0.3);
    const y = 2.0;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y, w: cw, h: 3.5, rectRadius: 0.1,
      fill: { color: C.white }, line: { color: C.border, width: 1 },
    });
    // icon 圈圈
    s.addShape(pres.shapes.OVAL, {
      x: x + 0.4, y: y + 0.4, w: 1.0, h: 1.0,
      fill: { color: "FDE0E1" }, line: { type: "none" },
    });
    s.addText(c.icon, {
      x: x + 0.4, y: y + 0.4, w: 1.0, h: 1.0,
      fontSize: 36, align: "center", valign: "middle", margin: 0,
    });
    s.addText(c.title, {
      x: x + 0.4, y: y + 1.6, w: cw - 0.8, h: 0.5,
      fontSize: 18, bold: true, color: C.navy, fontFace: FONT_TITLE, margin: 0,
    });
    s.addText(c.desc, {
      x: x + 0.4, y: y + 2.15, w: cw - 0.8, h: 1.2,
      fontSize: 12, color: C.ink, fontFace: FONT_BODY, margin: 0, paraSpaceAfter: 4,
    });
  });

  // 底部研究目標
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.6, y: 5.8, w: 12.1, h: 1.0, rectRadius: 0.08,
    fill: { color: C.navy }, line: { type: "none" },
  });
  s.addText("🎯 研究目標", {
    x: 0.9, y: 5.9, w: 2, h: 0.4,
    fontSize: 14, bold: true, color: C.gold, fontFace: FONT_TITLE, margin: 0,
  });
  s.addText("建立一個融合多種 ML 模型、可解釋、可重現的世界盃預測系統，並透過真實歷史數據驗證準確率優於隨機猜測 (33.3%)。", {
    x: 0.9, y: 6.25, w: 11.5, h: 0.5,
    fontSize: 12, color: C.cream, fontFace: FONT_BODY, margin: 0,
  });

  addFooter(s, 3, TOTAL);
}

// SLIDE 4: 研究目標與主要貢獻
{
  const s = pres.addSlide();
  s.background = { color: C.cream };
  addSectionLabel(s, "PART 01");
  addTitle(s, "研究目標與主要貢獻", "本專題的 5 大原創貢獻");

  const contribs = [
    { num: "01", title: "四層融合模型架構",       desc: "XGBoost 分類 + Dixon-Coles Poisson + 多因素 λ 修正 + MAP 比分" },
    { num: "02", title: "修正 Dixon-Coles 公式 bug",  desc: "原公式方向反轉導致強防守隊變成被預測輸球，修正後預測一致性大幅提升" },
    { num: "03", title: "對稱預測消除主場偏差",   desc: "正向＋反向各預測一次取平均，符合中性場地的世界盃實際情況" },
    { num: "04", title: "Walk-Forward 滾動驗證", desc: "用 2010 → 2014 → 2018 → 2022 漸進驗證，避免資訊洩漏" },
    { num: "05", title: "互動式 Streamlit 視覺化", desc: "7 頁完整應用，含 PCA 分群、Monte Carlo 模擬、費雪檢定" },
  ];
  contribs.forEach((c, i) => {
    const y = 2.0 + i * 0.9;
    // 編號徽章
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.6, y, w: 0.7, h: 0.7,
      fill: { color: C.navy }, line: { type: "none" },
    });
    s.addText(c.num, {
      x: 0.6, y, w: 0.7, h: 0.7,
      fontSize: 18, bold: true, color: C.gold, align: "center", valign: "middle",
      fontFace: FONT_TITLE, margin: 0,
    });
    // 標題
    s.addText(c.title, {
      x: 1.5, y: y + 0.05, w: 11, h: 0.35,
      fontSize: 15, bold: true, color: C.navy, fontFace: FONT_TITLE, margin: 0,
    });
    // 描述
    s.addText(c.desc, {
      x: 1.5, y: y + 0.38, w: 11, h: 0.32,
      fontSize: 11, color: C.muted, fontFace: FONT_BODY, margin: 0,
    });
  });

  addFooter(s, 4, TOTAL);
}

// ═════════════════════════════════════════════════════════════════
// PART 2: 資料工程
// ═════════════════════════════════════════════════════════════════

// SLIDE 5: 三大資料源
{
  const s = pres.addSlide();
  s.background = { color: C.cream };
  addSectionLabel(s, "PART 02");
  addTitle(s, "三大資料源", "公開資料庫 + 自建陣容資料的混合策略");

  const sources = [
    {
      num: "49,328", unit: "場",
      title: "國際足球比賽",
      desc: "Kaggle 公開資料集\n1872-2026 完整歷史\n含友誼賽、資格賽、世界盃",
    },
    {
      num: "67,894", unit: "筆",
      title: "FIFA 世界排名",
      desc: "FIFA 官方歷史月度排名\n用於 walk-forward 驗證的時點積分",
    },
    {
      num: "240", unit: "位",
      title: "主將陣容資料",
      desc: "自建 48 隊 × 5 主將\n含 OVR / 速度 / 射門 / 傳球等 7 屬性",
    },
  ];
  const cw = 3.95;
  sources.forEach((src, i) => {
    const x = 0.6 + i * (cw + 0.3);
    const y = 2.0;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y, w: cw, h: 3.8, rectRadius: 0.1,
      fill: { color: C.white }, line: { color: C.border, width: 1 },
    });
    // 大數字
    s.addText(src.num, {
      x: x + 0.3, y: y + 0.4, w: cw - 0.6, h: 1.2,
      fontSize: 56, bold: true, color: C.primary,
      fontFace: FONT_TITLE, align: "center", margin: 0,
    });
    s.addText(src.unit, {
      x: x + 0.3, y: y + 1.55, w: cw - 0.6, h: 0.3,
      fontSize: 13, color: C.muted, align: "center",
      fontFace: FONT_BODY, margin: 0,
    });
    // 標題
    s.addText(src.title, {
      x: x + 0.3, y: y + 2.0, w: cw - 0.6, h: 0.4,
      fontSize: 16, bold: true, color: C.navy, align: "center",
      fontFace: FONT_TITLE, margin: 0,
    });
    // 短分隔線
    s.addShape(pres.shapes.LINE, {
      x: x + cw / 2 - 0.3, y: y + 2.5, w: 0.6, h: 0,
      line: { color: C.gold, width: 2 },
    });
    // 描述
    s.addText(src.desc, {
      x: x + 0.3, y: y + 2.65, w: cw - 0.6, h: 1.0,
      fontSize: 11, color: C.ink, fontFace: FONT_BODY, align: "center",
      paraSpaceAfter: 4, margin: 0,
    });
  });

  // 底部說明
  s.addText("📌 資料時間範圍跨越 154 年，涵蓋 235 個國家隊，是公開可得最完整的足球資料集之一", {
    x: 0.6, y: 6.2, w: 12.1, h: 0.5,
    fontSize: 12, color: C.muted, italic: true, fontFace: FONT_BODY, align: "center",
  });

  addFooter(s, 5, TOTAL);
}

// SLIDE 6: 資料清洗與特徵工程
{
  const s = pres.addSlide();
  s.background = { color: C.cream };
  addSectionLabel(s, "PART 02");
  addTitle(s, "資料清洗與特徵工程", "從原始資料到模型可用特徵的 5 步驟");

  // 左：流程步驟
  const steps = [
    { num: "01", title: "資料清洗",       desc: "過濾缺失值 · 統一國家名稱 · 解析賽事類型" },
    { num: "02", title: "時間衰減權重",   desc: "exp(-0.1 × 年數) 讓近期比賽影響更大" },
    { num: "03", title: "球隊強度計算",   desc: "近 8 年加權勝率 / 進球 / 失球" },
    { num: "04", title: "FIFA 歷史對齊",  desc: "用該屆 WC 開幕前最近月份的 FIFA 積分" },
    { num: "05", title: "特徵向量化",     desc: "差值化（team1 - team2）+ 對稱性處理" },
  ];
  steps.forEach((st, i) => {
    const y = 2.0 + i * 0.85;
    s.addShape(pres.shapes.OVAL, {
      x: 0.7, y, w: 0.5, h: 0.5,
      fill: { color: C.primary }, line: { type: "none" },
    });
    s.addText(st.num, {
      x: 0.7, y, w: 0.5, h: 0.5,
      fontSize: 11, bold: true, color: C.white, align: "center", valign: "middle",
      fontFace: FONT_TITLE, margin: 0,
    });
    s.addText(st.title, {
      x: 1.4, y: y + 0.02, w: 5, h: 0.3,
      fontSize: 14, bold: true, color: C.navy, fontFace: FONT_TITLE, margin: 0,
    });
    s.addText(st.desc, {
      x: 1.4, y: y + 0.32, w: 5, h: 0.28,
      fontSize: 10, color: C.muted, fontFace: FONT_BODY, margin: 0,
    });
  });

  // 右：時間衰減權重表
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 7.0, y: 2.0, w: 5.7, h: 4.3, rectRadius: 0.1,
    fill: { color: C.white }, line: { color: C.border, width: 1 },
  });
  s.addText("⏰ 時間衰減權重表", {
    x: 7.3, y: 2.2, w: 5.1, h: 0.4,
    fontSize: 14, bold: true, color: C.navy, fontFace: FONT_TITLE, margin: 0,
  });
  s.addText("weight = exp(-0.1 × 年數)", {
    x: 7.3, y: 2.6, w: 5.1, h: 0.3,
    fontSize: 11, italic: true, color: C.muted, fontFace: FONT_SERIF, margin: 0,
  });
  // 表格
  const decayTable = [
    [{ text: "年數前", options: { bold: true, color: C.navy } },
     { text: "權重",   options: { bold: true, color: C.navy } },
     { text: "意義",   options: { bold: true, color: C.navy } }],
    ["1 年", "0.90", "近一年比賽幾乎滿權重"],
    ["3 年", "0.74", "前年/上屆 WC 仍重要"],
    ["5 年", "0.61", "中期歷史，影響約 60%"],
    ["8 年", "0.45", "上上屆 WC 約半權重"],
    ["10 年", "0.37", "更久遠的影響大幅折減"],
  ];
  s.addTable(decayTable, {
    x: 7.3, y: 3.1, w: 5.1, h: 3.0,
    colW: [1.1, 1.0, 3.0],
    fontSize: 11, fontFace: FONT_BODY,
    border: { type: "solid", pt: 0.5, color: C.border },
    color: C.ink,
    fill: { color: C.cream },
  });

  addFooter(s, 6, TOTAL);
}

// ═════════════════════════════════════════════════════════════════
// PART 3: 四層融合模型架構
// ═════════════════════════════════════════════════════════════════

// SLIDE 7: 模型整體架構（流程圖）
{
  const s = pres.addSlide();
  s.background = { color: C.cream };
  addSectionLabel(s, "PART 03");
  addTitle(s, "四層融合模型架構", "從原始特徵到最終比分預測的完整流程");

  // 4 個層的橫向流程圖
  const layers = [
    { num: "1", title: "XGBoost",        weight: "20%", desc: "勝/平/負三向分類" },
    { num: "2", title: "Dixon-Coles",    weight: "80%", desc: "Poisson 期望進球 λ" },
    { num: "3", title: "λ 多因素修正",   weight: "—",   desc: "主將 + FIFA + 狀態 + 經驗" },
    { num: "4", title: "MAP + MC",       weight: "—",   desc: "比分預測 + 10K 次模擬" },
  ];
  const lw = 2.85;
  const lgap = 0.15;
  layers.forEach((l, i) => {
    const x = 0.5 + i * (lw + lgap);
    const y = 2.2;
    // 卡片
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y, w: lw, h: 3.0, rectRadius: 0.08,
      fill: { color: C.white }, line: { color: C.border, width: 1 },
    });
    // 上方深底
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y, w: lw, h: 0.8, rectRadius: 0.08,
      fill: { color: C.navy }, line: { type: "none" },
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: y + 0.5, w: lw, h: 0.3,
      fill: { color: C.navy }, line: { type: "none" },
    });
    // 層號
    s.addText(`Layer ${l.num}`, {
      x, y: y + 0.1, w: lw, h: 0.3,
      fontSize: 11, color: C.gold, bold: true, align: "center",
      fontFace: FONT_BODY, charSpacing: 2, margin: 0,
    });
    s.addText(l.title, {
      x, y: y + 0.42, w: lw, h: 0.4,
      fontSize: 18, color: C.white, bold: true, align: "center",
      fontFace: FONT_TITLE, margin: 0,
    });
    // 權重大字
    s.addText(l.weight, {
      x, y: y + 1.0, w: lw, h: 0.9,
      fontSize: 48, bold: true, color: C.primary, align: "center",
      fontFace: FONT_TITLE, margin: 0,
    });
    s.addText("外層融合權重", {
      x, y: y + 1.95, w: lw, h: 0.25,
      fontSize: 9, color: C.muted, align: "center", fontFace: FONT_BODY, margin: 0,
    });
    // 說明
    s.addText(l.desc, {
      x: x + 0.2, y: y + 2.3, w: lw - 0.4, h: 0.6,
      fontSize: 11, color: C.ink, align: "center", fontFace: FONT_BODY, margin: 0,
    });
    // 箭頭（除最後一個）
    if (i < layers.length - 1) {
      s.addShape(pres.shapes.RIGHT_TRIANGLE, {
        x: x + lw + 0.005, y: y + 1.4, w: 0.14, h: 0.18,
        fill: { color: C.primary }, line: { type: "none" },
        rotate: 90,
      });
    }
  });

  // 底部公式
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.5, y: 5.7, w: 12.3, h: 1.2, rectRadius: 0.08,
    fill: { color: C.navy }, line: { type: "none" },
  });
  s.addText("最終機率公式", {
    x: 0.8, y: 5.8, w: 4, h: 0.35,
    fontSize: 12, color: C.gold, bold: true, fontFace: FONT_BODY, margin: 0,
  });
  s.addText("P(勝/平/負) = 0.20 × XGBoost + 0.80 × Poisson(λ)", {
    x: 0.8, y: 6.15, w: 12, h: 0.4,
    fontSize: 18, color: C.white, bold: true, fontFace: FONT_SERIF, margin: 0,
  });
  s.addText("λ = λ_DC × rank_factor × squad_factor × form_factor × experience_factor", {
    x: 0.8, y: 6.55, w: 12, h: 0.3,
    fontSize: 12, color: C.cream, italic: true, fontFace: FONT_SERIF, margin: 0,
  });

  addFooter(s, 7, TOTAL);
}

// SLIDE 8: Layer 1 XGBoost
{
  const s = pres.addSlide();
  s.background = { color: C.cream };
  addSectionLabel(s, "PART 03");
  addTitle(s, "Layer 1：XGBoost 分類器（20%）", "勝/平/負三向機率分類 · Walk-Forward 訓練");

  // 左：核心功能
  s.addText("🤖 核心角色", {
    x: 0.6, y: 2.0, w: 6, h: 0.4,
    fontSize: 16, bold: true, color: C.navy, fontFace: FONT_TITLE, margin: 0,
  });
  s.addText([
    { text: "直接輸出 P(勝) / P(平) / P(負) 三向機率\n", options: { breakLine: true } },
    { text: "200 棵樹 · max_depth = 4 · learning_rate = 0.05\n", options: { breakLine: true } },
    { text: "正向＋反向各預測一次取平均 → 消除主場偏差", options: {} },
  ], {
    x: 0.6, y: 2.45, w: 6, h: 1.5,
    fontSize: 12, color: C.ink, fontFace: FONT_BODY, paraSpaceAfter: 6, margin: 0,
  });

  s.addText("📊 輸入特徵（11 維）", {
    x: 0.6, y: 4.1, w: 6, h: 0.4,
    fontSize: 16, bold: true, color: C.navy, fontFace: FONT_TITLE, margin: 0,
  });
  const feats = [
    "pts_diff", "rank_diff", "win_rate_diff", "avg_goals_diff",
    "defense_diff", "form_diff", "experience", "goals_product",
    "confed_bonus", "recent_form", "knockout_exp",
  ];
  // 用 chip 形式
  feats.forEach((f, i) => {
    const col = i % 3;
    const row = Math.floor(i / 3);
    const x = 0.6 + col * 2.0;
    const y = 4.6 + row * 0.45;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y, w: 1.9, h: 0.35, rectRadius: 0.05,
      fill: { color: "FDE0E1" }, line: { color: C.primary, width: 0.5 },
    });
    s.addText(f, {
      x, y, w: 1.9, h: 0.35,
      fontSize: 10, color: C.navy, align: "center", valign: "middle",
      fontFace: "Consolas", bold: true, margin: 0,
    });
  });

  // 右：訓練流程
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 7.0, y: 2.0, w: 5.7, h: 4.5, rectRadius: 0.1,
    fill: { color: C.white }, line: { color: C.border, width: 1 },
  });
  s.addText("⚙️ Walk-Forward 訓練流程", {
    x: 7.3, y: 2.2, w: 5.1, h: 0.4,
    fontSize: 14, bold: true, color: C.navy, fontFace: FONT_TITLE, margin: 0,
  });
  // 訓練 → 驗證 step
  const trainSteps = [
    { train: "1990-2009", val: "WC 2010", acc: "53.3%" },
    { train: "1990-2013", val: "WC 2014", acc: "48.4%" },
    { train: "1990-2017", val: "WC 2018", acc: "50.0%" },
    { train: "1990-2021", val: "WC 2022", acc: "56.2%" },
  ];
  trainSteps.forEach((st, i) => {
    const y = 2.8 + i * 0.55;
    s.addText(`訓練 ${st.train}`, {
      x: 7.3, y, w: 2.0, h: 0.35,
      fontSize: 10, color: C.muted, fontFace: FONT_BODY, margin: 0,
    });
    s.addText("→", {
      x: 9.3, y, w: 0.3, h: 0.35,
      fontSize: 12, color: C.primary, bold: true, align: "center", margin: 0,
    });
    s.addText(`驗證 ${st.val}`, {
      x: 9.6, y, w: 1.5, h: 0.35,
      fontSize: 10, color: C.muted, fontFace: FONT_BODY, margin: 0,
    });
    s.addText(st.acc, {
      x: 11.2, y, w: 1.2, h: 0.35,
      fontSize: 12, bold: true, color: C.green, align: "right",
      fontFace: FONT_TITLE, margin: 0,
    });
  });
  // 平均
  s.addShape(pres.shapes.LINE, {
    x: 7.3, y: 5.15, w: 5.1, h: 0,
    line: { color: C.border, width: 1 },
  });
  s.addText("平均驗證準確率", {
    x: 7.3, y: 5.3, w: 3, h: 0.4,
    fontSize: 12, bold: true, color: C.navy, fontFace: FONT_TITLE, margin: 0,
  });
  s.addText("52.0%", {
    x: 10.0, y: 5.2, w: 2.4, h: 0.5,
    fontSize: 28, bold: true, color: C.primary, align: "right",
    fontFace: FONT_TITLE, margin: 0,
  });
  s.addText("vs 隨機猜測 33.3% · 提升 +18.7%", {
    x: 7.3, y: 5.85, w: 5.1, h: 0.4,
    fontSize: 11, italic: true, color: C.green, fontFace: FONT_BODY, margin: 0,
  });

  addFooter(s, 8, TOTAL);
}

// SLIDE 9: Layer 2 Dixon-Coles
{
  const s = pres.addSlide();
  s.background = { color: C.cream };
  addSectionLabel(s, "PART 03");
  addTitle(s, "Layer 2：Dixon-Coles Poisson（80%）", "從攻防強度推導期望進球，再積分出 W/D/L 機率");

  // 中央大公式
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.6, y: 2.0, w: 12.1, h: 1.5, rectRadius: 0.1,
    fill: { color: C.navy }, line: { type: "none" },
  });
  s.addText("核心公式", {
    x: 0.9, y: 2.15, w: 3, h: 0.3,
    fontSize: 11, color: C.gold, bold: true, charSpacing: 2,
    fontFace: FONT_BODY, margin: 0,
  });
  s.addText("λ_A = atk_A × vul_B / μ", {
    x: 0.9, y: 2.5, w: 11.2, h: 0.6,
    fontSize: 32, bold: true, color: C.white, align: "center",
    fontFace: FONT_SERIF, italic: true, margin: 0,
  });
  s.addText("atk_A: A 隊進攻強度 · vul_B: B 隊防守脆弱度 · μ: 國際賽進球均值 (1.35)", {
    x: 0.9, y: 3.15, w: 11.2, h: 0.3,
    fontSize: 12, color: C.cream, align: "center", italic: true,
    fontFace: FONT_BODY, margin: 0,
  });

  // 三個盒子：atk、vul、足聯校正
  const formulas = [
    { title: "atk (進攻強度)", formula: "raw_goals × cs", desc: "弱聯盟進攻打折" },
    { title: "vul (防守脆弱度)", formula: "raw_conceded ÷ √cs", desc: "弱聯盟防守實際更弱" },
    { title: "cs (足聯品質係數)", formula: "0.72 ~ 1.00", desc: "OFC 0.72 → CONMEBOL 1.00" },
  ];
  formulas.forEach((f, i) => {
    const x = 0.6 + i * 4.0;
    const y = 3.9;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y, w: 3.85, h: 1.5, rectRadius: 0.08,
      fill: { color: C.white }, line: { color: C.border, width: 1 },
    });
    s.addText(f.title, {
      x: x + 0.2, y: y + 0.15, w: 3.65, h: 0.35,
      fontSize: 13, bold: true, color: C.navy, fontFace: FONT_TITLE, margin: 0,
    });
    s.addText(f.formula, {
      x: x + 0.2, y: y + 0.55, w: 3.65, h: 0.4,
      fontSize: 16, color: C.primary, bold: true, fontFace: FONT_SERIF, margin: 0,
    });
    s.addText(f.desc, {
      x: x + 0.2, y: y + 1.0, w: 3.65, h: 0.4,
      fontSize: 11, color: C.muted, italic: true, fontFace: FONT_BODY, margin: 0,
    });
  });

  // 底部：從 λ 到 W/D/L
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.6, y: 5.7, w: 12.1, h: 1.2, rectRadius: 0.08,
    fill: { color: "FDE9A8" }, line: { color: C.gold, width: 1 },
  });
  s.addText("📐 從 λ 計算 W/D/L 機率", {
    x: 0.85, y: 5.85, w: 6, h: 0.35,
    fontSize: 13, bold: true, color: C.navy, fontFace: FONT_TITLE, margin: 0,
  });
  s.addText("P(W) = Σ Poisson(g₁ | λ_A) × Poisson(g₂ | λ_B)   其中 g₁ > g₂", {
    x: 0.85, y: 6.2, w: 11.5, h: 0.35,
    fontSize: 13, color: C.ink, fontFace: FONT_SERIF, italic: true, margin: 0,
  });
  s.addText("枚舉 0~7 進球範圍涵蓋 99.9%+ 機率，即可得勝/平/負完整分布", {
    x: 0.85, y: 6.55, w: 11.5, h: 0.3,
    fontSize: 11, color: C.muted, italic: true, fontFace: FONT_BODY, margin: 0,
  });

  addFooter(s, 9, TOTAL);
}

// SLIDE 10: Layer 3 λ 多因素修正
{
  const s = pres.addSlide();
  s.background = { color: C.cream };
  addSectionLabel(s, "PART 03");
  addTitle(s, "Layer 3：λ 多因素複合修正", "在純 DC 基礎上疊加 4 個現實因子");

  // 公式
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.6, y: 2.0, w: 12.1, h: 0.9, rectRadius: 0.08,
    fill: { color: C.navy }, line: { type: "none" },
  });
  s.addText("λ_final = λ_DC × rank_factor × squad_factor × form_factor × experience_factor", {
    x: 0.6, y: 2.1, w: 12.1, h: 0.65,
    fontSize: 17, bold: true, color: C.white, align: "center", valign: "middle",
    fontFace: FONT_SERIF, italic: true, margin: 0,
  });

  // 4 個 factor 表格
  const factorsTable = [
    [{ text: "因素",        options: { bold: true, color: C.white, fill: { color: C.navy } } },
     { text: "公式",        options: { bold: true, color: C.white, fill: { color: C.navy } } },
     { text: "次方",        options: { bold: true, color: C.white, fill: { color: C.navy } } },
     { text: "意義",        options: { bold: true, color: C.white, fill: { color: C.navy } } }],
    [
      { text: "🥇 主將 OVR" },
      { text: "(ovr₁ / ovr₂)^0.35", options: { fontFace: "Consolas", color: C.primary } },
      { text: "0.35", options: { bold: true, color: C.primary, align: "center" } },
      { text: "陣容明星密度（權重最大）" },
    ],
    [
      { text: "🥈 FIFA 積分" },
      { text: "((p₁+300)/(p₂+300))^0.22", options: { fontFace: "Consolas", color: C.primary } },
      { text: "0.22", options: { bold: true, color: C.primary, align: "center" } },
      { text: "歷史排名差距" },
    ],
    [
      { text: "🥉 近 2 年勝率" },
      { text: "((f₁+0.3)/(f₂+0.3))^0.18", options: { fontFace: "Consolas", color: C.primary } },
      { text: "0.18", options: { bold: true, color: C.primary, align: "center" } },
      { text: "球隊近期狀態" },
    ],
    [
      { text: "🏅 淘汰賽經驗" },
      { text: "((e₁+10)/(e₂+10))^0.08", options: { fontFace: "Consolas", color: C.primary } },
      { text: "0.08", options: { bold: true, color: C.primary, align: "center" } },
      { text: "大賽 DNA（影響最小）" },
    ],
  ];
  s.addTable(factorsTable, {
    x: 0.6, y: 3.1, w: 12.1, h: 2.5,
    colW: [2.5, 3.8, 1.3, 4.5],
    fontSize: 12, fontFace: FONT_BODY,
    border: { type: "solid", pt: 0.5, color: C.border },
    color: C.ink, valign: "middle",
  });

  // 底部範例
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.6, y: 5.85, w: 12.1, h: 1.0, rectRadius: 0.08,
    fill: { color: "FEF1C9" }, line: { color: C.gold, width: 1 },
  });
  s.addText("💡 範例", {
    x: 0.85, y: 5.95, w: 1.5, h: 0.35,
    fontSize: 12, bold: true, color: C.navy, fontFace: FONT_TITLE, margin: 0,
  });
  s.addText("主將平均 OVR 85 vs 79：squad_factor = (85/79)^0.35 ≈ 1.027 → λ 提升 2.7%", {
    x: 0.85, y: 6.25, w: 11.5, h: 0.55,
    fontSize: 13, color: C.ink, fontFace: FONT_SERIF, italic: true, margin: 0,
  });

  addFooter(s, 10, TOTAL);
}

// SLIDE 11: Layer 4 MAP + Monte Carlo
{
  const s = pres.addSlide();
  s.background = { color: C.cream };
  addSectionLabel(s, "PART 03");
  addTitle(s, "Layer 4：MAP 比分 + Monte Carlo 模擬", "從機率到具體比分，再到完整賽程模擬");

  // 左：MAP 比分
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.6, y: 2.0, w: 6.0, h: 4.8, rectRadius: 0.1,
    fill: { color: C.white }, line: { color: C.border, width: 1 },
  });
  s.addText("🎯 MAP 最高機率比分", {
    x: 0.9, y: 2.2, w: 5.4, h: 0.4,
    fontSize: 16, bold: true, color: C.navy, fontFace: FONT_TITLE, margin: 0,
  });
  s.addText("Maximum A Posteriori — 找出機率分布中最高的單一比分", {
    x: 0.9, y: 2.6, w: 5.4, h: 0.35,
    fontSize: 11, color: C.muted, italic: true, fontFace: FONT_BODY, margin: 0,
  });
  // 3 步驟
  const mapSteps = [
    { num: "1", text: "依融合機率 argmax(W,D,L) 取主預測方向" },
    { num: "2", text: "在該方向 cells 中找最高 Poisson 機率" },
    { num: "3", text: "標記為預測比分 + ★ 顯示在矩陣" },
  ];
  mapSteps.forEach((st, i) => {
    const y = 3.2 + i * 0.65;
    s.addShape(pres.shapes.OVAL, {
      x: 0.9, y, w: 0.4, h: 0.4,
      fill: { color: C.primary }, line: { type: "none" },
    });
    s.addText(st.num, {
      x: 0.9, y, w: 0.4, h: 0.4,
      fontSize: 11, bold: true, color: C.white, align: "center", valign: "middle",
      fontFace: FONT_TITLE, margin: 0,
    });
    s.addText(st.text, {
      x: 1.4, y: y + 0.05, w: 4.9, h: 0.35,
      fontSize: 12, color: C.ink, fontFace: FONT_BODY, margin: 0,
    });
  });
  // 重點
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.9, y: 5.4, w: 5.4, h: 1.2, rectRadius: 0.05,
    fill: { color: "E4ECE5" }, line: { color: C.green, width: 1 },
  });
  s.addText("✅ 解決平局悖論", {
    x: 1.1, y: 5.5, w: 5, h: 0.35,
    fontSize: 12, bold: true, color: C.green, fontFace: FONT_TITLE, margin: 0,
  });
  s.addText("確保預測比分方向永遠與機率方向一致，避免「平局機率 20%卻每場都被預測平局」", {
    x: 1.1, y: 5.85, w: 5, h: 0.7,
    fontSize: 11, color: C.ink, fontFace: FONT_BODY, margin: 0,
  });

  // 右：Monte Carlo
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 6.8, y: 2.0, w: 6.0, h: 4.8, rectRadius: 0.1,
    fill: { color: C.white }, line: { color: C.border, width: 1 },
  });
  s.addText("🎲 Monte Carlo 賽程模擬", {
    x: 7.1, y: 2.2, w: 5.4, h: 0.4,
    fontSize: 16, bold: true, color: C.navy, fontFace: FONT_TITLE, margin: 0,
  });
  s.addText("10,000 次完整賽程模擬，統計奪冠機率", {
    x: 7.1, y: 2.6, w: 5.4, h: 0.35,
    fontSize: 11, color: C.muted, italic: true, fontFace: FONT_BODY, margin: 0,
  });
  // 流程
  const mcSteps = [
    "12 組小組賽 → 各組前 2 出線",
    "8 個最佳第 3 名晉級 → 32 強",
    "R32 → R16 → QF → SF → Final",
    "累計奪冠次數 / 10,000 = 奪冠機率",
  ];
  mcSteps.forEach((t, i) => {
    const y = 3.2 + i * 0.45;
    s.addText("▶", {
      x: 7.1, y, w: 0.3, h: 0.3,
      fontSize: 11, color: C.primary, bold: true, margin: 0,
    });
    s.addText(t, {
      x: 7.4, y, w: 5.1, h: 0.3,
      fontSize: 12, color: C.ink, fontFace: FONT_BODY, margin: 0,
    });
  });
  // 統計亮點
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 7.1, y: 5.2, w: 5.4, h: 1.45, rectRadius: 0.05,
    fill: { color: "FDD8D9" }, line: { color: C.primary, width: 1 },
  });
  s.addText("10,000", {
    x: 7.1, y: 5.3, w: 5.4, h: 0.6,
    fontSize: 36, bold: true, color: C.primary, align: "center",
    fontFace: FONT_TITLE, margin: 0,
  });
  s.addText("次模擬 · 約 1,040,000 場虛擬比賽", {
    x: 7.1, y: 5.95, w: 5.4, h: 0.3,
    fontSize: 11, color: C.muted, align: "center", italic: true, fontFace: FONT_BODY, margin: 0,
  });
  s.addText("（每次完整賽程 = 104 場）", {
    x: 7.1, y: 6.25, w: 5.4, h: 0.3,
    fontSize: 10, color: C.muted, align: "center", fontFace: FONT_BODY, margin: 0,
  });

  addFooter(s, 11, TOTAL);
}

// ═════════════════════════════════════════════════════════════════
// PART 4: 模型驗證
// ═════════════════════════════════════════════════════════════════

// SLIDE 12: Walk-Forward 驗證
{
  const s = pres.addSlide();
  s.background = { color: C.cream };
  addSectionLabel(s, "PART 04");
  addTitle(s, "Walk-Forward 滾動驗證", "用時間順序模擬「過去訓練、未來預測」的真實場景");

  // 概念說明
  s.addText("📌 為什麼需要 Walk-Forward？", {
    x: 0.6, y: 2.0, w: 12, h: 0.4,
    fontSize: 16, bold: true, color: C.navy, fontFace: FONT_TITLE, margin: 0,
  });
  s.addText("傳統 train/test split 會「未來看到過去」造成資料洩漏。Walk-Forward 保證每一輪驗證時模型只看得到該時點之前的資料，符合實際預測場景。", {
    x: 0.6, y: 2.4, w: 12, h: 0.7,
    fontSize: 12, color: C.ink, fontFace: FONT_BODY, margin: 0,
  });

  // 4 輪驗證的視覺化（時間軸）
  const rounds = [
    { train: "1990-2009", val: "WC 2010", acc: 53.3 },
    { train: "1990-2013", val: "WC 2014", acc: 48.4 },
    { train: "1990-2017", val: "WC 2018", acc: 50.0 },
    { train: "1990-2021", val: "WC 2022", acc: 56.2 },
  ];
  const tlY = 3.5;
  rounds.forEach((r, i) => {
    const y = tlY + i * 0.7;
    // 訓練段
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.6, y, w: 7.5, h: 0.5,
      fill: { color: "7A859E" }, line: { type: "none" },
    });
    s.addText(`訓練 ${r.train}`, {
      x: 0.6, y, w: 7.5, h: 0.5,
      fontSize: 11, color: C.white, align: "center", valign: "middle", bold: true,
      fontFace: FONT_BODY, margin: 0,
    });
    // 驗證段
    s.addShape(pres.shapes.RECTANGLE, {
      x: 8.15, y, w: 2.0, h: 0.5,
      fill: { color: C.primary }, line: { type: "none" },
    });
    s.addText(`驗證 ${r.val}`, {
      x: 8.15, y, w: 2.0, h: 0.5,
      fontSize: 11, color: C.white, align: "center", valign: "middle", bold: true,
      fontFace: FONT_BODY, margin: 0,
    });
    // 準確率
    s.addText(`${r.acc.toFixed(1)}%`, {
      x: 10.5, y, w: 1.5, h: 0.5,
      fontSize: 18, bold: true, color: C.green, align: "right", valign: "middle",
      fontFace: FONT_TITLE, margin: 0,
    });
    s.addText("acc", {
      x: 12.0, y, w: 0.5, h: 0.5,
      fontSize: 9, color: C.muted, valign: "middle", fontFace: FONT_BODY, margin: 0,
    });
  });

  // 底部總結
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.6, y: 6.4, w: 12.1, h: 0.6, rectRadius: 0.05,
    fill: { color: "E4ECE5" }, line: { color: C.green, width: 1 },
  });
  s.addText([
    { text: "✅ 平均驗證準確率 ", options: { bold: true, color: C.green } },
    { text: "52.0%", options: { bold: true, color: C.green, fontSize: 18 } },
    { text: "    vs 隨機猜測基準 33.3%    →    ", options: { color: C.muted } },
    { text: "顯著優於隨機 +18.7 個百分點", options: { bold: true, color: C.navy } },
  ], {
    x: 0.85, y: 6.4, w: 11.6, h: 0.6,
    fontSize: 12, fontFace: FONT_BODY, valign: "middle", margin: 0,
  });

  addFooter(s, 12, TOTAL);
}

// SLIDE 13: 混淆矩陣分析
{
  const s = pres.addSlide();
  s.background = { color: C.cream };
  addSectionLabel(s, "PART 04");
  addTitle(s, "混淆矩陣：模型每個類別的表現", "用 2022 卡達世界盃 64 場小組賽當測試集");

  // 左：3x3 混淆矩陣
  s.addText("混淆矩陣（2022 WC · 64 場）", {
    x: 0.6, y: 2.0, w: 6, h: 0.4,
    fontSize: 14, bold: true, color: C.navy, fontFace: FONT_TITLE, margin: 0,
  });
  // 表格
  const cm = [
    [{ text: "",          options: { fill: { color: C.cream }, bold: true } },
     { text: "預測：負",  options: { fill: { color: C.navy }, color: C.white, bold: true, align: "center" } },
     { text: "預測：平",  options: { fill: { color: C.navy }, color: C.white, bold: true, align: "center" } },
     { text: "預測：勝",  options: { fill: { color: C.navy }, color: C.white, bold: true, align: "center" } }],
    [{ text: "實際：負",  options: { fill: { color: C.navy }, color: C.white, bold: true, align: "center" } },
     { text: "12", options: { fill: { color: C.green }, color: C.white, bold: true, align: "center", fontSize: 18 } },
     { text: "2",  options: { align: "center" } },
     { text: "7",  options: { align: "center" } }],
    [{ text: "實際：平",  options: { fill: { color: C.navy }, color: C.white, bold: true, align: "center" } },
     { text: "5",  options: { align: "center" } },
     { text: "1",  options: { fill: { color: "95B097" }, color: C.white, bold: true, align: "center", fontSize: 18 } },
     { text: "9",  options: { align: "center" } }],
    [{ text: "實際：勝",  options: { fill: { color: C.navy }, color: C.white, bold: true, align: "center" } },
     { text: "5",  options: { align: "center" } },
     { text: "1",  options: { align: "center" } },
     { text: "22", options: { fill: { color: C.green }, color: C.white, bold: true, align: "center", fontSize: 18 } }],
  ];
  s.addTable(cm, {
    x: 0.6, y: 2.5, w: 6.2, h: 3.6,
    colW: [1.4, 1.6, 1.6, 1.6], rowH: 0.7,
    fontSize: 13, fontFace: FONT_BODY,
    border: { type: "solid", pt: 0.5, color: C.border },
    color: C.ink, valign: "middle",
  });

  // 右：三個 Recall 卡
  const recalls = [
    { label: "主隊勝 Recall", val: "79%", note: "強隊優勢明顯易預測", color: C.green },
    { label: "主隊負 Recall", val: "57%", note: "弱隊輸球可預期", color: C.gold },
    { label: "平局 Recall",   val: "7%",  note: "⚠️ 最難預測（業界共同難題）", color: C.primary },
  ];
  recalls.forEach((r, i) => {
    const y = 2.0 + i * 1.5;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 7.2, y, w: 5.5, h: 1.3, rectRadius: 0.08,
      fill: { color: C.white }, line: { color: r.color, width: 1.5 },
    });
    s.addText(r.label, {
      x: 7.4, y: y + 0.15, w: 3.5, h: 0.4,
      fontSize: 13, bold: true, color: C.navy, fontFace: FONT_TITLE, margin: 0,
    });
    s.addText(r.val, {
      x: 7.4, y: y + 0.55, w: 3.5, h: 0.6,
      fontSize: 36, bold: true, color: r.color, fontFace: FONT_TITLE, margin: 0,
    });
    s.addText(r.note, {
      x: 11.0, y: y + 0.35, w: 1.6, h: 0.8,
      fontSize: 9, color: C.muted, italic: true, align: "right",
      fontFace: FONT_BODY, valign: "middle", margin: 0,
    });
  });

  // 底部
  s.addText("📌 整體準確率 54.7%，遠優於隨機 33.3%。平局是預測難題，所有商業博弈公司皆有此問題。", {
    x: 0.6, y: 6.55, w: 12.1, h: 0.4,
    fontSize: 11, italic: true, color: C.muted, align: "center", fontFace: FONT_BODY,
  });

  addFooter(s, 13, TOTAL);
}

// SLIDE 14: ROC + AUC + 校準
{
  const s = pres.addSlide();
  s.background = { color: C.cream };
  addSectionLabel(s, "PART 04");
  addTitle(s, "ROC 曲線 + 校準診斷", "兩個進階指標：排序能力 vs 機率可信度");

  // 左：ROC AUC
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.6, y: 2.0, w: 6.0, h: 4.5, rectRadius: 0.1,
    fill: { color: C.white }, line: { color: C.border, width: 1 },
  });
  s.addText("📈 ROC AUC — 排序能力", {
    x: 0.9, y: 2.2, w: 5.4, h: 0.4,
    fontSize: 14, bold: true, color: C.navy, fontFace: FONT_TITLE, margin: 0,
  });
  s.addText("「模型把該贏的場次排得越前面，AUC 越高」", {
    x: 0.9, y: 2.6, w: 5.4, h: 0.3,
    fontSize: 10, italic: true, color: C.muted, fontFace: FONT_BODY, margin: 0,
  });
  // AUC bars
  const aucs = [
    { name: "主隊勝", val: 0.78, color: C.green, rating: "🟢 良好" },
    { name: "主隊負", val: 0.75, color: C.gold,  rating: "🟢 良好" },
    { name: "平局",   val: 0.51, color: C.muted, rating: "🔴 偏弱" },
  ];
  aucs.forEach((a, i) => {
    const y = 3.1 + i * 0.95;
    s.addText(a.name, {
      x: 0.9, y, w: 1.5, h: 0.4,
      fontSize: 13, bold: true, color: C.navy, fontFace: FONT_TITLE, margin: 0,
    });
    s.addText(a.val.toFixed(2), {
      x: 2.5, y, w: 1.0, h: 0.4,
      fontSize: 20, bold: true, color: a.color, fontFace: FONT_TITLE, margin: 0,
    });
    s.addText(a.rating, {
      x: 3.7, y: y + 0.05, w: 2.7, h: 0.3,
      fontSize: 11, color: C.muted, fontFace: FONT_BODY, margin: 0,
    });
    // 進度條
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.9, y: y + 0.45, w: 5.4, h: 0.18,
      fill: { color: C.border }, line: { type: "none" },
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.9, y: y + 0.45, w: 5.4 * a.val, h: 0.18,
      fill: { color: a.color }, line: { type: "none" },
    });
  });

  // 右：校準
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 6.8, y: 2.0, w: 6.0, h: 4.5, rectRadius: 0.1,
    fill: { color: C.white }, line: { color: C.border, width: 1 },
  });
  s.addText("🎚 校準誤差 (ECE)", {
    x: 7.1, y: 2.2, w: 5.4, h: 0.4,
    fontSize: 14, bold: true, color: C.navy, fontFace: FONT_TITLE, margin: 0,
  });
  s.addText("「模型說 70% 勝率時，是否真的 70% 場次主隊贏」", {
    x: 7.1, y: 2.6, w: 5.4, h: 0.3,
    fontSize: 10, italic: true, color: C.muted, fontFace: FONT_BODY, margin: 0,
  });
  // ECE 大值
  s.addText("0.083", {
    x: 7.1, y: 3.0, w: 5.4, h: 1.0,
    fontSize: 56, bold: true, color: C.green, fontFace: FONT_TITLE, align: "center", margin: 0,
  });
  s.addText("ECE (Expected Calibration Error)", {
    x: 7.1, y: 4.05, w: 5.4, h: 0.3,
    fontSize: 11, color: C.muted, italic: true, align: "center",
    fontFace: FONT_BODY, margin: 0,
  });
  s.addShape(pres.shapes.LINE, {
    x: 7.3, y: 4.5, w: 5.0, h: 0,
    line: { color: C.border, width: 0.5 },
  });
  s.addText("🟢 < 0.05 校準極佳   🟡 < 0.10 良好   🟠 ≥ 0.10 偏差大", {
    x: 7.1, y: 4.7, w: 5.4, h: 0.35,
    fontSize: 10, color: C.muted, align: "center", fontFace: FONT_BODY, margin: 0,
  });
  s.addText("👉 本模型 ECE = 0.083 屬「良好」，機率值可按字面信賴", {
    x: 7.1, y: 5.1, w: 5.4, h: 0.6,
    fontSize: 11, color: C.ink, bold: true, italic: true,
    align: "center", fontFace: FONT_BODY, margin: 0,
  });
  s.addText("→ Monte Carlo 10,000 次模擬結果可信", {
    x: 7.1, y: 5.65, w: 5.4, h: 0.5,
    fontSize: 11, color: C.green, italic: true, align: "center",
    fontFace: FONT_BODY, margin: 0,
  });

  // 底部
  s.addText("📐 ROC 量「排序能力」，校準量「機率本身的可信度」— 兩個指標檢驗不同面向", {
    x: 0.6, y: 6.6, w: 12.1, h: 0.4,
    fontSize: 11, italic: true, color: C.muted, align: "center", fontFace: FONT_BODY,
  });

  addFooter(s, 14, TOTAL);
}

// SLIDE 15: 費雪精確檢定
{
  const s = pres.addSlide();
  s.background = { color: C.cream };
  addSectionLabel(s, "PART 04");
  addTitle(s, "費雪精確檢定 (Fisher's Exact Test)", "用嚴格統計方法證明模型不是運氣好");

  // 問題
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.6, y: 2.0, w: 12.1, h: 0.9, rectRadius: 0.08,
    fill: { color: "FEF1C9" }, line: { color: C.gold, width: 1 },
  });
  s.addText("❓ 問題：54.7% 正確率是真有預測力，還是純粹運氣？", {
    x: 0.85, y: 2.1, w: 11.5, h: 0.4,
    fontSize: 14, bold: true, color: C.navy, fontFace: FONT_TITLE, margin: 0,
  });
  s.addText("用「一對多 (One-vs-Rest)」拆 3×3 為 2×2 列聯表，對每類做費雪檢定", {
    x: 0.85, y: 2.5, w: 11.5, h: 0.35,
    fontSize: 11, color: C.ink, italic: true, fontFace: FONT_BODY, margin: 0,
  });

  // 三類結果
  const fisher = [
    [
      { text: "類別",            options: { bold: true, color: C.white, fill: { color: C.navy } } },
      { text: "p-value",         options: { bold: true, color: C.white, fill: { color: C.navy }, align: "center" } },
      { text: "Odds Ratio",      options: { bold: true, color: C.white, fill: { color: C.navy }, align: "center" } },
      { text: "顯著性 (α=0.05)", options: { bold: true, color: C.white, fill: { color: C.navy }, align: "center" } },
    ],
    [
      { text: "主隊負" },
      { text: "0.0014", options: { align: "center", bold: true, color: C.green } },
      { text: "5.20",   options: { align: "center" } },
      { text: "✅ 極顯著 ✓✓✓",  options: { align: "center", color: C.green, bold: true } },
    ],
    [
      { text: "平局" },
      { text: "0.6234", options: { align: "center", color: C.muted } },
      { text: "1.08",   options: { align: "center" } },
      { text: "⚠️ 未達顯著",     options: { align: "center", color: C.muted } },
    ],
    [
      { text: "主隊勝" },
      { text: "0.0001", options: { align: "center", bold: true, color: C.green } },
      { text: "8.94",   options: { align: "center" } },
      { text: "✅ 極顯著 ✓✓✓",  options: { align: "center", color: C.green, bold: true } },
    ],
  ];
  s.addTable(fisher, {
    x: 0.6, y: 3.1, w: 12.1, h: 2.0,
    colW: [2.0, 2.5, 2.5, 5.1], rowH: 0.5,
    fontSize: 12, fontFace: FONT_BODY,
    border: { type: "solid", pt: 0.5, color: C.border },
    color: C.ink, valign: "middle",
  });

  // 整體卡方
  s.addText("📊 整體 3×3 卡方檢定（χ² test of independence）", {
    x: 0.6, y: 5.4, w: 12.1, h: 0.35,
    fontSize: 13, bold: true, color: C.navy, fontFace: FONT_TITLE,
  });
  const chi2items = [
    { label: "χ² 統計量", val: "20.8" },
    { label: "自由度",    val: "4" },
    { label: "p-value",   val: "< 0.001" },
    { label: "結論",      val: "✅ 顯著" },
  ];
  chi2items.forEach((c, i) => {
    const x = 0.6 + i * 3.05;
    const y = 5.85;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y, w: 2.9, h: 0.95, rectRadius: 0.08,
      fill: { color: C.white }, line: { color: C.border, width: 1 },
    });
    s.addText(c.label, {
      x, y: y + 0.1, w: 2.9, h: 0.3,
      fontSize: 11, color: C.muted, align: "center", fontFace: FONT_BODY, margin: 0,
    });
    s.addText(c.val, {
      x, y: y + 0.4, w: 2.9, h: 0.45,
      fontSize: 18, bold: true, color: C.primary, align: "center",
      fontFace: FONT_TITLE, margin: 0,
    });
  });

  addFooter(s, 15, TOTAL);
}

// ═════════════════════════════════════════════════════════════════
// PART 5: 球隊風格分群
// ═════════════════════════════════════════════════════════════════

// SLIDE 16: K-Means 分群結果
{
  const s = pres.addSlide();
  s.background = { color: C.cream };
  addSectionLabel(s, "PART 05");
  addTitle(s, "球隊風格分群 — K-Means + PCA", "用非監督式學習找出 48 隊的天然分類");

  // 演算法說明
  s.addText("🔬 方法：K-Means (k=3) + PCA 二維降維 + Silhouette 0.31", {
    x: 0.6, y: 2.0, w: 12, h: 0.4,
    fontSize: 13, italic: true, color: C.muted, fontFace: FONT_BODY,
  });

  // 三類風格
  const styles = [
    {
      icon: "⚡", name: "攻擊型", count: 15, color: C.primary,
      desc: "場均進球高 · 主動進攻",
      teams: "Argentina · Brazil · France · Germany · England · Spain · Belgium · Portugal · Netherlands · Morocco · Norway · Iran · Japan · Senegal · Algeria",
    },
    {
      icon: "⚖️", name: "平衡型", count: 27, color: C.gold,
      desc: "攻守均衡 · 進失球中等",
      teams: "Australia · Austria · Canada · Cape Verde · Colombia · Croatia · DR Congo · Ecuador · Egypt · Ghana · Haiti · Iraq · Ivory Coast · Jordan · Mexico · New Zealand · Panama · Qatar · Saudi Arabia · Scotland · South Africa · South Korea · Sweden · Switzerland · Tunisia · Uruguay · Uzbekistan",
    },
    {
      icon: "🛡️", name: "防守型", count: 6, color: C.blue,
      desc: "場均失球低 · 鐵桶陣",
      teams: "Bosnia and Herzegovina · Curacao · Czechia · Paraguay · Turkiye · USA",
    },
  ];
  // 預混淺色（25% 飽和度於白底）
  const styleTints = { [C.primary]: "FBC1C4", [C.gold]: "FCE3A4", [C.blue]: "C6D9E0" };
  styles.forEach((st, i) => {
    const y = 2.5 + i * 1.55;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 0.6, y, w: 12.1, h: 1.45, rectRadius: 0.08,
      fill: { color: C.white }, line: { color: st.color, width: 1.5 },
    });
    // icon 圈
    s.addShape(pres.shapes.OVAL, {
      x: 0.85, y: y + 0.3, w: 0.85, h: 0.85,
      fill: { color: styleTints[st.color] || "EEEEEE" }, line: { type: "none" },
    });
    s.addText(st.icon, {
      x: 0.85, y: y + 0.3, w: 0.85, h: 0.85,
      fontSize: 32, align: "center", valign: "middle", margin: 0,
    });
    // 名稱 + 隊數
    s.addText(st.name, {
      x: 1.85, y: y + 0.2, w: 3, h: 0.4,
      fontSize: 18, bold: true, color: st.color, fontFace: FONT_TITLE, margin: 0,
    });
    s.addText(`${st.count} 隊`, {
      x: 1.85, y: y + 0.6, w: 3, h: 0.3,
      fontSize: 11, color: C.muted, fontFace: FONT_BODY, margin: 0,
    });
    s.addText(st.desc, {
      x: 1.85, y: y + 0.95, w: 3, h: 0.35,
      fontSize: 11, italic: true, color: C.ink, fontFace: FONT_BODY, margin: 0,
    });
    // 球隊清單
    s.addText(st.teams, {
      x: 5.0, y: y + 0.2, w: 7.6, h: 1.1,
      fontSize: 9, color: C.muted, fontFace: FONT_BODY, valign: "top", margin: 0,
    });
  });

  addFooter(s, 16, TOTAL);
}

// ═════════════════════════════════════════════════════════════════
// PART 6: 奪冠機率預測
// ═════════════════════════════════════════════════════════════════

// SLIDE 17: Monte Carlo 奪冠 Top 10
{
  const s = pres.addSlide();
  s.background = { color: C.cream };
  addSectionLabel(s, "PART 06");
  addTitle(s, "奪冠機率 Top 10", "10,000 次完整賽程模擬的累計結果");

  const champs = [
    { rank: 1, team: "🇦🇷 阿根廷",     en: "Argentina",    champ: 6.14, final: 18.37, grp: 77.6 },
    { rank: 2, team: "🇧🇷 巴西",       en: "Brazil",       champ: 6.01, final: 18.79, grp: 77.5 },
    { rank: 3, team: "🏴󠁧󠁢󠁥󠁮󠁧󠁿 英格蘭",     en: "England",      champ: 5.80, final: 18.10, grp: 76.2 },
    { rank: 4, team: "🇫🇷 法國",       en: "France",       champ: 5.75, final: 18.59, grp: 76.8 },
    { rank: 5, team: "🇪🇸 西班牙",     en: "Spain",        champ: 5.01, final: 16.67, grp: 76.0 },
    { rank: 6, team: "🇧🇪 比利時",     en: "Belgium",      champ: 4.99, final: 16.35, grp: 76.5 },
    { rank: 7, team: "🇵🇹 葡萄牙",     en: "Portugal",     champ: 4.65, final: 15.57, grp: 75.8 },
    { rank: 8, team: "🇳🇱 荷蘭",       en: "Netherlands",  champ: 4.62, final: 15.55, grp: 72.8 },
    { rank: 9, team: "🇩🇪 德國",       en: "Germany",      champ: 4.44, final: 15.76, grp: 76.2 },
    { rank: 10, team: "🇨🇴 哥倫比亞",  en: "Colombia",     champ: 3.41, final: 12.22, grp: 62.4 },
  ];

  const table = [
    [
      { text: "排名", options: { bold: true, color: C.white, fill: { color: C.navy }, align: "center" } },
      { text: "球隊", options: { bold: true, color: C.white, fill: { color: C.navy } } },
      { text: "奪冠機率", options: { bold: true, color: C.white, fill: { color: C.navy }, align: "center" } },
      { text: "進決賽機率", options: { bold: true, color: C.white, fill: { color: C.navy }, align: "center" } },
      { text: "小組第一機率", options: { bold: true, color: C.white, fill: { color: C.navy }, align: "center" } },
    ],
    ...champs.map(c => [
      { text: c.rank === 1 ? "🥇" : c.rank === 2 ? "🥈" : c.rank === 3 ? "🥉" : `${c.rank}`,
        options: { align: "center", bold: c.rank <= 3, fontSize: c.rank <= 3 ? 16 : 12 } },
      { text: `${c.team}   ${c.en}`, options: { bold: c.rank <= 3 } },
      { text: `${c.champ.toFixed(2)}%`, options: { align: "center", bold: true,
                                                    color: c.rank <= 3 ? C.primary : C.ink } },
      { text: `${c.final.toFixed(1)}%`, options: { align: "center" } },
      { text: `${c.grp.toFixed(1)}%`, options: { align: "center" } },
    ]),
  ];
  s.addTable(table, {
    x: 0.6, y: 2.0, w: 12.1, h: 4.6,
    colW: [0.9, 4.0, 2.4, 2.4, 2.4], rowH: 0.42,
    fontSize: 12, fontFace: FONT_BODY,
    border: { type: "solid", pt: 0.5, color: C.border },
    color: C.ink, valign: "middle",
  });

  // 底部說明
  s.addText("📌 奪冠機率分散：Top 10 加起來才 ~48%，符合世界盃高度不確定性的本質。", {
    x: 0.6, y: 6.75, w: 12.1, h: 0.3,
    fontSize: 11, italic: true, color: C.muted, align: "center", fontFace: FONT_BODY,
  });

  addFooter(s, 17, TOTAL);
}

// SLIDE 18: 奪冠機率分布圖（用 chart）
{
  const s = pres.addSlide();
  s.background = { color: C.cream };
  addSectionLabel(s, "PART 06");
  addTitle(s, "奪冠機率分布視覺化", "前 15 名球隊的奪冠機率長條圖");

  const data = [{
    name: "奪冠機率 (%)",
    labels: ["阿根廷", "巴西", "英格蘭", "法國", "西班牙", "比利時", "葡萄牙",
             "荷蘭", "德國", "哥倫比亞", "烏拉圭", "克羅埃西亞", "摩洛哥", "墨西哥", "瑞士"],
    values: [6.14, 6.01, 5.80, 5.75, 5.01, 4.99, 4.65, 4.62, 4.44, 3.41, 3.39, 3.30, 2.89, 2.78, 2.57],
  }];
  s.addChart(pres.charts.BAR, data, {
    x: 0.6, y: 2.0, w: 12.1, h: 4.5,
    barDir: "bar",
    chartColors: [C.primary],
    showLegend: false,
    catAxisLabelFontSize: 11,
    valAxisLabelFontSize: 10,
    showValue: true,
    dataLabelFontSize: 10,
    dataLabelColor: C.navy,
    dataLabelFormatCode: "0.00\"%\"",
    valAxisMinVal: 0,
    valAxisMaxVal: 7,
    valGridLine: { color: C.border, style: "solid", size: 0.5 },
    catGridLine: { style: "none" },
  });

  s.addText("💡 觀察：歐洲傳統強權（法/英/比/西/德/荷/葡）佔 7 席，南美兩強（巴/阿）穩居前二", {
    x: 0.6, y: 6.6, w: 12.1, h: 0.4,
    fontSize: 11, italic: true, color: C.muted, align: "center", fontFace: FONT_BODY,
  });

  addFooter(s, 18, TOTAL);
}

// ═════════════════════════════════════════════════════════════════
// PART 7: 技術創新與挑戰
// ═════════════════════════════════════════════════════════════════

// SLIDE 19: 修正 Dixon-Coles 公式 bug
{
  const s = pres.addSlide();
  s.background = { color: C.cream };
  addSectionLabel(s, "PART 07");
  addTitle(s, "🔧 技術挑戰 #1：Dixon-Coles 公式 bug", "強防守隊反而被預測輸球的根本原因");

  // 問題描述
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.6, y: 2.0, w: 12.1, h: 0.9, rectRadius: 0.08,
    fill: { color: "FDE0E1" }, line: { color: C.primary, width: 1 },
  });
  s.addText("🐛 問題：南韓三項數據全優（勝率/進球/失球），卻被預測輸 1-2 給南非", {
    x: 0.85, y: 2.1, w: 11.5, h: 0.35,
    fontSize: 14, bold: true, color: C.primary, fontFace: FONT_TITLE, margin: 0,
  });
  s.addText("→ λ_KR < λ_SA，方向完全相反", {
    x: 0.85, y: 2.5, w: 11.5, h: 0.35,
    fontSize: 12, color: C.ink, italic: true, fontFace: FONT_BODY, margin: 0,
  });

  // 對比舊 vs 新公式
  const cmp = [
    { type: "舊 (錯誤)", formula: "λ_A = atk_A × LEAGUE_AVG / def_B",
      logic: "對手失球少 → 我的 λ 變大\n（直覺反！）", color: C.primary },
    { type: "新 (正確)", formula: "λ_A = atk_A × vul_B / LEAGUE_AVG",
      logic: "對手失球多 (vul 高) → 我容易進球\n（合直覺）", color: C.green },
  ];
  cmp.forEach((c, i) => {
    const x = 0.6 + i * 6.1;
    const y = 3.1;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y, w: 6.0, h: 2.5, rectRadius: 0.1,
      fill: { color: C.white }, line: { color: c.color, width: 2 },
    });
    s.addText(c.type, {
      x: x + 0.3, y: y + 0.2, w: 5.4, h: 0.4,
      fontSize: 15, bold: true, color: c.color, fontFace: FONT_TITLE, margin: 0,
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: x + 0.3, y: y + 0.7, w: 5.4, h: 0.7,
      fill: { color: C.navy }, line: { type: "none" },
    });
    s.addText(c.formula, {
      x: x + 0.3, y: y + 0.7, w: 5.4, h: 0.7,
      fontSize: 14, color: C.gold, bold: true, align: "center", valign: "middle",
      fontFace: FONT_SERIF, italic: true, margin: 0,
    });
    s.addText(c.logic, {
      x: x + 0.3, y: y + 1.55, w: 5.4, h: 0.8,
      fontSize: 12, color: C.ink, italic: true, fontFace: FONT_BODY, margin: 0,
    });
  });

  // 影響評估
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.6, y: 5.85, w: 12.1, h: 1.0, rectRadius: 0.08,
    fill: { color: "FEF1C9" }, line: { color: C.gold, width: 1 },
  });
  s.addText("✅ 修正後驗算", {
    x: 0.85, y: 5.95, w: 4, h: 0.35,
    fontSize: 13, bold: true, color: C.navy, fontFace: FONT_TITLE, margin: 0,
  });
  s.addText("KR vs CZ: λ_KR=1.87 > λ_CZ=0.66 → MAP 1-0    |    KR vs SA: λ_KR=1.19 > λ_SA=0.67 → MAP 1-0", {
    x: 0.85, y: 6.3, w: 11.5, h: 0.5,
    fontSize: 12, color: C.green, bold: true, italic: true, fontFace: FONT_SERIF, margin: 0,
  });

  addFooter(s, 19, TOTAL);
}

// SLIDE 20: 平局悖論解法
{
  const s = pres.addSlide();
  s.background = { color: C.cream };
  addSectionLabel(s, "PART 07");
  addTitle(s, "🔧 技術挑戰 #2：平局悖論", "「平局機率只 20%」為何「每場都被預測平局」？");

  // 解釋
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.6, y: 2.0, w: 12.1, h: 1.2, rectRadius: 0.08,
    fill: { color: "FEF1C9" }, line: { color: C.gold, width: 1 },
  });
  s.addText("🎯 看似矛盾的觀察", {
    x: 0.85, y: 2.1, w: 11.5, h: 0.35,
    fontSize: 14, bold: true, color: C.navy, fontFace: FONT_TITLE, margin: 0,
  });
  s.addText("P(平局) 顯示只 20-27%，但 MAP「最高機率單一比分」常是 (1,1) — 為什麼？", {
    x: 0.85, y: 2.5, w: 11.5, h: 0.35,
    fontSize: 12, color: C.ink, italic: true, fontFace: FONT_BODY, margin: 0,
  });
  s.addText("答：當 λ_A ≈ λ_B 接近時，Poisson 分布的最高單一格落在 (1,1)，但「整體勝方總和」可能仍 > 平局", {
    x: 0.85, y: 2.85, w: 11.5, h: 0.35,
    fontSize: 11, color: C.muted, fontFace: FONT_BODY, margin: 0,
  });

  // 對比兩種策略
  const tabs = [
    {
      title: "❌ 舊策略：絕對 MAP",
      method: "找整個 Poisson 矩陣最大值",
      problem: "λ 接近時恆挑 (1,1) 平局\n與機率方向不一致",
      color: C.primary,
    },
    {
      title: "✅ 新策略：方向約束 MAP",
      method: "先決定主預測方向 → 在該方向 cells 找最強",
      problem: "比分方向 = 機率方向\n保證內部一致",
      color: C.green,
    },
  ];
  tabs.forEach((t, i) => {
    const x = 0.6 + i * 6.1;
    const y = 3.45;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y, w: 6.0, h: 2.6, rectRadius: 0.1,
      fill: { color: C.white }, line: { color: t.color, width: 2 },
    });
    s.addText(t.title, {
      x: x + 0.3, y: y + 0.2, w: 5.4, h: 0.4,
      fontSize: 14, bold: true, color: t.color, fontFace: FONT_TITLE, margin: 0,
    });
    s.addText("方法：", {
      x: x + 0.3, y: y + 0.7, w: 1.2, h: 0.3,
      fontSize: 11, bold: true, color: C.navy, fontFace: FONT_BODY, margin: 0,
    });
    s.addText(t.method, {
      x: x + 1.4, y: y + 0.7, w: 4.3, h: 0.6,
      fontSize: 11, color: C.ink, fontFace: FONT_BODY, margin: 0,
    });
    s.addText("結果：", {
      x: x + 0.3, y: y + 1.5, w: 1.2, h: 0.3,
      fontSize: 11, bold: true, color: C.navy, fontFace: FONT_BODY, margin: 0,
    });
    s.addText(t.problem, {
      x: x + 1.4, y: y + 1.5, w: 4.3, h: 0.9,
      fontSize: 11, color: C.ink, italic: true, fontFace: FONT_BODY, margin: 0,
    });
  });

  // 結論
  s.addText("📌 預測比分 = 與整體勝率方向相符的「最強單一比分」，避免「預測平局又說對方贏面大」的視覺矛盾", {
    x: 0.6, y: 6.5, w: 12.1, h: 0.4,
    fontSize: 11, italic: true, color: C.muted, align: "center", fontFace: FONT_BODY,
  });

  addFooter(s, 20, TOTAL);
}

// SLIDE 21: 資料品質校正
{
  const s = pres.addSlide();
  s.background = { color: C.cream };
  addSectionLabel(s, "PART 07");
  addTitle(s, "🔧 技術挑戰 #3：資料品質校正", "用程式審計揪出 + 修正 48 隊 × 240 球員的資料錯誤");

  // 兩欄：TEAM_INFO + SQUAD_DATA
  const fixes = [
    {
      icon: "📊", title: "TEAM_INFO 校正", count: "48 隊 / 18 項修正",
      color: C.primary,
      items: [
        "Argentina 從 #3 改為 #1（2022 WC 冠軍）",
        "3 對重複 rank 修正 (Switzerland/Czechia/SA)",
        "Bosnia 50→78（差 28 名）",
        "Norway/Canada/Algeria 等 10+ 隊校正",
        "rank ↔ pts 100% 嚴格反向單調",
      ],
    },
    {
      icon: "👥", title: "SQUAD_DATA 校正", count: "240 球員 / 9 項修正",
      color: C.blue,
      items: [
        "7 位退役/退出國家隊球員替換",
        "Modrić Real Madrid → AC Milan（2025）",
        "Džeko Fenerbahçe → Başakşehir",
        "Messi 37→38 歲、Ronaldo 40→41 歲",
        "G. Ramos (葡萄牙人) 從 Cape Verde 移除",
      ],
    },
  ];
  fixes.forEach((f, i) => {
    const x = 0.6 + i * 6.1;
    const y = 2.0;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y, w: 6.0, h: 4.8, rectRadius: 0.1,
      fill: { color: C.white }, line: { color: f.color, width: 1.5 },
    });
    // 標題
    s.addText(f.icon, {
      x: x + 0.3, y: y + 0.2, w: 0.6, h: 0.6,
      fontSize: 30, margin: 0,
    });
    s.addText(f.title, {
      x: x + 1.0, y: y + 0.25, w: 4.8, h: 0.4,
      fontSize: 16, bold: true, color: C.navy, fontFace: FONT_TITLE, margin: 0,
    });
    s.addText(f.count, {
      x: x + 1.0, y: y + 0.65, w: 4.8, h: 0.3,
      fontSize: 11, color: f.color, bold: true, fontFace: FONT_BODY, margin: 0,
    });
    // items
    f.items.forEach((item, j) => {
      const iy = y + 1.2 + j * 0.6;
      s.addText("▶", {
        x: x + 0.3, y: iy, w: 0.3, h: 0.4,
        fontSize: 12, color: f.color, bold: true, margin: 0,
      });
      s.addText(item, {
        x: x + 0.65, y: iy, w: 5.1, h: 0.4,
        fontSize: 11, color: C.ink, fontFace: FONT_BODY, valign: "top", margin: 0,
      });
    });
  });

  addFooter(s, 21, TOTAL);
}

// ═════════════════════════════════════════════════════════════════
// PART 8: 結論
// ═════════════════════════════════════════════════════════════════

// SLIDE 22: 主要發現
{
  const s = pres.addSlide();
  s.background = { color: C.cream };
  addSectionLabel(s, "PART 08");
  addTitle(s, "主要發現", "本專題的 4 個關鍵成果");

  const findings = [
    {
      num: "01", color: C.primary,
      title: "模型準確率顯著優於隨機",
      detail: "Walk-Forward 平均 52.0%，比隨機 33.3% 高 18.7 個百分點。費雪檢定 + 卡方檢定都拒絕「模型在亂猜」的虛無假設。",
    },
    {
      num: "02", color: C.gold,
      title: "Top 10 奪冠機率分散合理",
      detail: "阿根廷 6.14% / 巴西 6.01% / 英格蘭 5.80% / 法國 5.75% — 前 10 累計約 50%。符合世界盃高度不確定性的本質，沒有出現單一球隊獨大的失真。",
    },
    {
      num: "03", color: C.green,
      title: "風格分群揭示 15 攻 27 平 6 守 結構",
      detail: "K-Means 自動發現的分群與直覺一致：歐美/南美強隊多為攻擊型，東歐/亞洲球隊多為平衡或防守型。Silhouette 0.33 顯示適度可分。",
    },
    {
      num: "04", color: C.blue,
      title: "公式修正 + 多因素 λ 帶來預測一致性",
      detail: "修正 Dixon-Coles 公式方向錯誤、引入 4 因素複合 λ 修正、加入方向約束 MAP，三項共同確保預測結果與球隊強度視覺一致。",
    },
  ];
  findings.forEach((f, i) => {
    const y = 2.0 + i * 1.2;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.6, y, w: 0.12, h: 1.05,
      fill: { color: f.color }, line: { type: "none" },
    });
    s.addText(f.num, {
      x: 0.85, y, w: 1.0, h: 0.5,
      fontSize: 32, bold: true, color: f.color, fontFace: FONT_TITLE, margin: 0,
    });
    s.addText(f.title, {
      x: 2.0, y: y + 0.05, w: 10.7, h: 0.4,
      fontSize: 16, bold: true, color: C.navy, fontFace: FONT_TITLE, margin: 0,
    });
    s.addText(f.detail, {
      x: 2.0, y: y + 0.45, w: 10.7, h: 0.65,
      fontSize: 11, color: C.ink, fontFace: FONT_BODY, margin: 0,
    });
  });

  addFooter(s, 22, TOTAL);
}

// SLIDE 23: 限制與未來工作
{
  const s = pres.addSlide();
  s.background = { color: C.cream };
  addSectionLabel(s, "PART 08");
  addTitle(s, "系統限制與未來改進", "誠實面對：本系統的 5 個侷限與後續方向");

  const items = [
    {
      icon: "⚠️", title: "平局預測能力有限",
      now: "平局 Recall 7%、AUC 0.51",
      future: "嘗試 Skellam 分布或專門平局模型，導入即時賠率回饋校正",
    },
    {
      icon: "📅", title: "球員傷病/紅黃牌未考慮",
      now: "用季前主將平均 OVR 估算實力",
      future: "整合即時傷兵名單 API（如 Transfermarkt），動態調整 OVR",
    },
    {
      icon: "🌐", title: "資料時間範圍可再擴張",
      now: "近 8 年加權，2025-2026 資料 1,167 場",
      future: "增加歐國盃、美洲盃、亞洲盃等大賽結果作為遷移學習來源",
    },
    {
      icon: "🎲", title: "Monte Carlo 假設賽程獨立",
      now: "假設每場勝率不受前場結果影響",
      future: "加入「momentum」、「疲勞」、「傷病累積」等動態狀態變數",
    },
    {
      icon: "🤖", title: "模型可解釋性可加強",
      now: "提供特徵重要性 + 費雪檢定",
      future: "整合 SHAP values 對單場預測做特徵級貢獻分解",
    },
  ];
  items.forEach((it, i) => {
    const y = 2.0 + i * 1.0;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 0.6, y, w: 12.1, h: 0.85, rectRadius: 0.08,
      fill: { color: C.white }, line: { color: C.border, width: 1 },
    });
    s.addText(it.icon, {
      x: 0.8, y, w: 0.6, h: 0.85,
      fontSize: 22, align: "center", valign: "middle", margin: 0,
    });
    s.addText(it.title, {
      x: 1.5, y: y + 0.1, w: 4.5, h: 0.35,
      fontSize: 13, bold: true, color: C.navy, fontFace: FONT_TITLE, margin: 0,
    });
    s.addText(it.now, {
      x: 1.5, y: y + 0.45, w: 4.5, h: 0.3,
      fontSize: 10, color: C.muted, italic: true, fontFace: FONT_BODY, margin: 0,
    });
    // 箭頭
    s.addText("→", {
      x: 6.1, y, w: 0.5, h: 0.85,
      fontSize: 18, color: C.primary, bold: true, align: "center", valign: "middle", margin: 0,
    });
    s.addText("未來方向", {
      x: 6.7, y: y + 0.05, w: 1.5, h: 0.3,
      fontSize: 10, color: C.primary, bold: true, fontFace: FONT_BODY, margin: 0,
    });
    s.addText(it.future, {
      x: 6.7, y: y + 0.32, w: 6.0, h: 0.5,
      fontSize: 11, color: C.ink, fontFace: FONT_BODY, margin: 0,
    });
  });

  addFooter(s, 23, TOTAL);
}

// SLIDE 24: 結尾 Thank You
{
  const s = pres.addSlide();
  s.background = { color: C.navy };

  // 背景裝飾線
  s.addShape(pres.shapes.LINE, {
    x: 1, y: 2.0, w: 4, h: 0,
    line: { color: C.gold, width: 3 },
  });

  s.addText("謝謝聆聽", {
    x: 1, y: 2.4, w: 11, h: 1.5,
    fontSize: 80, bold: true, color: C.white, fontFace: FONT_TITLE, margin: 0,
  });
  s.addText("Thank You & Questions", {
    x: 1, y: 3.9, w: 11, h: 0.6,
    fontSize: 24, color: C.primary, italic: true, fontFace: FONT_SERIF, margin: 0,
  });

  // 線上 Demo 提示
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 1, y: 5.0, w: 11.3, h: 1.5, rectRadius: 0.1,
    fill: { color: "2D3E69" }, line: { color: C.gold, width: 1 },
  });
  s.addText("🌐 線上互動 Demo", {
    x: 1.3, y: 5.15, w: 11, h: 0.4,
    fontSize: 14, bold: true, color: C.gold, fontFace: FONT_BODY, margin: 0,
  });
  s.addText("https://worldcup-ml-2026-bzplzw7hoy7g5dcizaakvt.streamlit.app/", {
    x: 1.3, y: 5.55, w: 11, h: 0.4,
    fontSize: 14, color: C.cream, fontFace: "Consolas", margin: 0,
  });
  s.addText("📦 原始碼：github.com/AstorYeh/worldcup-ml-2026", {
    x: 1.3, y: 6.0, w: 11, h: 0.4,
    fontSize: 11, color: C.cream, italic: true, fontFace: FONT_BODY, margin: 0,
  });

  // 底部標籤
  s.addText("FINAL PROJECT · 2026", {
    x: 1, y: 6.95, w: 11.3, h: 0.3,
    fontSize: 10, color: C.gold, charSpacing: 4, bold: true,
    align: "center", fontFace: FONT_BODY, margin: 0,
  });
}

// ── 輸出 ──
pres.writeFile({ fileName: "2026_WorldCup_ML_期末報告.pptx" })
  .then(name => console.log(`✅ PPT 產生完成: ${name}`))
  .catch(err => { console.error("產生失敗:", err); process.exit(1); });
