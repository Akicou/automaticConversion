/**
 * Theme Toggle Functionality
 * Handles switching between dark and light themes with localStorage persistence
 */

class ThemeManager {
    constructor() {
        this.currentTheme = this.getStoredTheme() || this.getSystemPreference();
        this.init();
    }

    init() {
        // Apply the initial theme
        this.applyTheme(this.currentTheme);

        // Listen for system theme changes
        if (window.matchMedia) {
            const darkModeQuery = window.matchMedia('(prefers-color-scheme: dark)');
            darkModeQuery.addEventListener('change', (e) => {
                // Only auto-switch if user hasn't manually set a preference
                if (!this.getStoredTheme()) {
                    this.currentTheme = e.matches ? 'dark' : 'light';
                    this.applyTheme(this.currentTheme);
                }
            });
        }
    }

    getStoredTheme() {
        return localStorage.getItem('theme');
    }

    getSystemPreference() {
        if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
            return 'dark';
        }
        return 'light';
    }

    applyTheme(theme) {
        if (theme === 'light') {
            document.documentElement.setAttribute('data-theme', 'light');
        } else {
            document.documentElement.removeAttribute('data-theme');
        }
        this.updateThemeButton(theme);
    }

    toggle() {
        this.currentTheme = this.currentTheme === 'dark' ? 'light' : 'dark';
        this.applyTheme(this.currentTheme);
        localStorage.setItem('theme', this.currentTheme);

        // Show toast notification
        if (typeof showToast !== 'undefined') {
            showToast('info', `Switched to ${this.currentTheme} theme`, { duration: 2000 });
        }
    }

    updateThemeButton(theme) {
        const button = document.getElementById('theme-toggle');
        if (button) {
            button.textContent = theme === 'dark' ? '☀️' : '🌙';
            button.setAttribute('aria-label', `Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`);
        }
    }
}

// Create global theme manager instance
const themeManager = new ThemeManager();

// Theme toggle function for button onclick
function toggleTheme() {
    themeManager.toggle();
}
