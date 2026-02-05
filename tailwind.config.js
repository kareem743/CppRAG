/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "rgb(2 6 23)", // slate-950
        surface: "rgb(15 23 42)", // slate-900
        border: "rgb(30 41 59)", // slate-800
        "accent-ai": "rgb(99 102 241)", // indigo-500
        "accent-success": "rgb(52 211 153)", // emerald-400
      },
      boxShadow: {
        "cyber-glow": "0 0 0 1px rgb(99 102 241 / 0.2), 0 0 24px rgb(99 102 241 / 0.25)",
      },
    },
  },
  plugins: [],
};
