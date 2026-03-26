/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "#0b0f17",
        surface: "#0f1624",
        primary: "#e5e7eb",
        border: "#223148",
        accent: "#38bdf8",
        ink: "#e5e7eb",
        muted: "#94a3b8",
        sand: "#0b1a2d",
        slate: "#0b1220",
        glow: "#22d3ee",
      },
    },
  },
  plugins: [],
};
