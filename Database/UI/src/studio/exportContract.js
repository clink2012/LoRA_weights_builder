// This reader never manufactures loader vectors from catalogue block profiles.
export function readLoaderExport(payload) {
  const value = payload?.loader_export;
  if (!value || value.status !== "ready" || !Array.isArray(value.slot_values) || !value.slot_values.length || typeof value.numeric_csv !== "string") {
    return { ready: false, reason: value?.reason || "A verified loader mapping is needed before these weights can be copied." };
  }
  const labels = value.architecture_slot_labels;
  const values = value.architecture_slot_values;
  const finiteNumbers = (list) => Array.isArray(list) && list.length > 0 && list.every((number) => typeof number === "number" && Number.isFinite(number));
  const validLabels = (list, count) => Array.isArray(list) && list.length === count && list.every((label) => typeof label === "string" && label.trim().length > 0);
  const fields = value.numeric_csv.split(",");
  const numericFields = fields.every((field) => /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(field.trim()));
  const complete = finiteNumbers(values) && finiteNumbers(value.slot_values)
    && validLabels(labels, values.length) && validLabels(value.slot_labels, value.slot_values.length)
    && value.architecture_slot_count === values.length && value.loader_slot_count === value.slot_values.length
    && numericFields && fields.length === value.slot_values.length
    && fields.every((field, index) => Number(field) === value.slot_values[index]);
  if (!complete) return { ready: false, reason: "The server returned an incomplete or inconsistent loader vector. Prepare the stack again before copying." };
  return { ...value, ready: true, csv: value.numeric_csv, contract_id: value.adapter_id,
    slots: labels.map((label, index) => ({ label, value: values[index], group: label.startsWith("DOUBLE") ? "Double blocks" : label.startsWith("SINGLE") ? "Single blocks" : "Base" })) };
}
