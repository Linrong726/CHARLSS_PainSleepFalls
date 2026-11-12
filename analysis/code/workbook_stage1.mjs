import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

const [root] = process.argv.slice(2);
const packageRoot = process.env.STAGE1_NODE_MODULES;
if (!packageRoot) throw new Error("Set STAGE1_NODE_MODULES to the bundled Node.js packages path.");
const entry = path.join(packageRoot, "@oai", "artifact-tool", "dist", "artifact_tool.mjs");
const { Workbook, SpreadsheetFile } = await import(pathToFileURL(entry).href);
const spec = JSON.parse(await fs.readFile(path.join(root, "run", "workbook_data.json"), "utf8"));
const workbook = Workbook.create();

function column(n) {
  let name = "";
  for (; n > 0; n = Math.floor((n - 1) / 26)) {
    name = String.fromCharCode(65 + ((n - 1) % 26)) + name;
  }
  return name;
}

for (const tab of spec) {
  const sheet = workbook.worksheets.add(tab.name);
  const end = column(tab.values[0].length);
  const data = sheet.getRange(`A1:${end}${tab.values.length}`);
  data.values = tab.values;
  data.format.font.name = "Calibri";
  data.format.font.size = 10;
  data.format.rowHeight = 17;
  const head = sheet.getRange(`A1:${end}1`);
  head.format.fill = "#183854";
  head.format.font.color = "#FFFFFF";
  head.format.font.bold = true;
  head.format.rowHeight = 24;
  sheet.freezePanes.freezeRows(1);
  for (const [i, header] of tab.values[0].entries()) {
    const col = column(i + 1);
    const longest = Math.max(...tab.values.map(row => String(row[i] ?? "").length));
    sheet.getRange(`${col}:${col}`).format.columnWidth = Math.max(11, Math.min(longest + 2, 66));
    const body = tab.values.length > 1 ? sheet.getRange(`${col}2:${col}${tab.values.length}`) : null;
    if (body && ["estimate", "ci_low", "ci_high", "beta", "se", "smd", "percent", "crude_proportion"].includes(header)) {
      body.setNumberFormat("0.000000");
    }
  }
}
const inspection = await workbook.inspect({
  kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 50 }, summary: "Workbook error scan"
});
await fs.writeFile(path.join(root, "run", "workbook_inspection.ndjson"), inspection.ndjson);
const image = await workbook.render({sheetName: "Results", range: "A1:J9", scale: 1.15});
await fs.writeFile(path.join(root, "run", "workbook_preview.png"), new Uint8Array(await image.arrayBuffer()));
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(path.join(root, "RESULTS_STAGE1.xlsx"));
