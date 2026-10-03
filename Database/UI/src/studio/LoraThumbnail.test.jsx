import { cleanup, fireEvent, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import LoraThumbnail from "./LoraThumbnail";

afterEach(cleanup);
describe("local catalogue thumbnails", () => {
  it("requests only the stable-id endpoint and reserves an accessible decorative image slot", () => {
    const { container } = render(<LoraThumbnail apiBase="/api" stableId="FLX-PPL/1" name="Portrait" />);
    const image = container.querySelector("img");
    expect(image.getAttribute("src")).toBe("/api/lora/FLX-PPL%2F1/thumbnail");
    expect(image.getAttribute("alt")).toBe("");
    expect(image.getAttribute("loading")).toBe("lazy");
    fireEvent.load(image);
    expect(container.textContent).toBe("");
  });
  it("retains the name initials when the local preview is missing or rejected", () => {
    const { container } = render(<LoraThumbnail apiBase="/api" stableId="FLX-PPL-1" name="Portrait" />);
    fireEvent.error(container.querySelector("img"));
    expect(container.querySelector("img")).toBeNull();
    expect(container.textContent).toBe("PO");
  });
  it("does not request external or unresolvable image endpoints", () => {
    const { container } = render(<LoraThumbnail apiBase="https://external.invalid/api" stableId="FLX-PPL-1" name="Portrait" />);
    expect(container.querySelector("img")).toBeNull();
    expect(container.textContent).toBe("PO");
  });
});
