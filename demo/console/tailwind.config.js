/** Design tokens from SketchRAG_Dashboard_Design_Specification.docx, section 3. */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        app: "#0B1120", deep: "#080D17", card: "#151F30", panel: "#0E1624", line: "#233248",
        teal: { DEFAULT: "#2DD4BF", soft: "#4CE0B3" }, retrieval: { DEFAULT: "#60A5FA", soft: "#70A8FF" },
        memory: { DEFAULT: "#A78BFA", soft: "#B39AFF" }, fallback: { DEFAULT: "#FBBF24", soft: "#FFC857" },
        story: "#F27FBA", ink: { DEFAULT: "#E6EDF6", muted: "#8A9AB2", dim: "#5D6B82" },
      },
      fontFamily: { sans: ["Inter", "system-ui", "sans-serif"], mono: ["JetBrains Mono", "ui-monospace", "monospace"] },
      borderRadius: { card: "12px", ctl: "8px" },
    },
  },
  plugins: [],
};
