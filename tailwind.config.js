/** @type {import('tailwindcss').Config} */
module.exports = {
  // The 'content' array is CRITICAL.
  // It lists all the files Tailwind must scan for classes (e.g., 'bg-red-500').
  content: [
    // Include your HTML file path
    "./templates/**/*.html",
    // Include any JavaScript/React/Angular files you might have
    "./src/**/*.{js,ts,jsx,tsx}",
    "./index.html",
  ],
  theme: {
    // This is where you customize colors, fonts, spacing, etc.
    extend: {
        // 1. Primary Background Color (Mapping --color-bg: #1f2937)
        'rb-bg': '#1f2937',

        // 2. Default Text Color (Mapping --color-text-default: #e5e7eb)
        'rb-text-default': '#e5e7eb',

        // 3. Accent/Button Color (Mapping --color-accent: #67e8f9)
        // This is used for the bright agent button.
        'rb-accent': '#67e8f9',

        // 4. Primary Focus/Outline Color (Mapping --color-primary: #a78bfa)
        // This is used for focus rings on inputs.
        'rb-primary': '#a78bfa',
    },
  },
  // Plugins for typography, forms, etc., go here
  plugins: [],
}
