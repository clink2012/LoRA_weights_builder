import { describe, expect, it } from "vitest";
import { readLoaderExport } from "./exportContract";

const contract = () => ({ status: "ready", numeric_csv: "1.0000,-0.1234", slot_labels: ["BASE", "DOUBLE 0"], slot_values: [1, -0.1234], architecture_slot_labels: ["BASE", "DOUBLE 0"], architecture_slot_values: [1, -0.1234], architecture_slot_count: 2, loader_slot_count: 2 });
describe("loader response boundary", () => {
  it("preserves server formatting and signed values", () => {
    const result = readLoaderExport({ loader_export: contract() });
    expect(result.ready).toBe(true);
    expect(result.csv).toBe("1.0000,-0.1234");
    expect(result.slots[1].value).toBe(-0.1234);
  });
  it.each([
    { numeric_csv: "1,0.9999" }, { numeric_csv: "1," }, { numeric_csv: "1,Infinity" },
    { slot_values: [1, NaN] }, { architecture_slot_values: [1, Infinity] },
    { slot_labels: ["BASE", null] }, { architecture_slot_labels: ["BASE", 3] },
    { architecture_slot_count: 58 }, { loader_slot_count: 58 }, { slot_values: ["1", "-0.1234"] },
  ])("blocks malformed ready payload %j", (change) => {
    expect(readLoaderExport({ loader_export: { ...contract(), ...change } }).ready).toBe(false);
  });
  it("never upgrades a legacy analysis vector into an export", () => {
    expect(readLoaderExport({ block_weights: [1, 0.5], block_weights_csv: "1,0.5" }).ready).toBe(false);
  });
});
