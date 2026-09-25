/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        darkbg: "#0b0f19",
        glasscard: "rgba(21, 30, 49, 0.75)",
      },
    },
  },
  plugins: [],
};
