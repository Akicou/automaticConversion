/**
 * Keyboard Navigation and Shortcuts
 * Provides keyboard shortcuts and enhanced accessibility
 */

class KeyboardManager {
    constructor() {
        this.shortcuts = {
            'ctrl+k': this.focusSearch.bind(this),
            'cmd+k': this.focusSearch.bind(this),
            'escape': this.handleEscape.bind(this),
            '?': this.showHelp.bind(this)
        };
        this.init();
    }

    init() {
        document.addEventListener('keydown', (e) => this.handleKeyDown(e));
        this.setupAccessibility();
    }

    handleKeyDown(e) {
        // Ignore if user is typing in an input field
        if (e.target.tagName === 'INPUT' ||
            e.target.tagName === 'TEXTAREA' ||
            e.target.isContentEditable) {
            // Allow Escape to exit input fields
            if (e.key === 'Escape') {
                e.target.blur();
            }
            return;
        }

        const key = this.getKeyString(e);
        const handler = this.shortcuts[key.toLowerCase()];

        if (handler) {
            e.preventDefault();
            handler(e);
        }
    }

    getKeyString(e) {
        const parts = [];
        if (e.ctrlKey) parts.push('ctrl');
        if (e.metaKey) parts.push('cmd');
        if (e.shiftKey) parts.push('shift');
        if (e.altKey) parts.push('alt');
        parts.push(e.key.toLowerCase());
        return parts.join('+');
    }

    focusSearch() {
        const searchInput = document.getElementById('modelSearch');
        if (searchInput) {
            searchInput.focus();
            showToast('info', 'Press Escape to exit search', { duration: 2000 });
        }
    }

    handleEscape() {
        // Close modals
        const modals = document.querySelectorAll('.modal[style*="display: block"]');
        modals.forEach(modal => {
            if (modal.style.display !== 'none') {
                modal.style.display = 'none';
            }
        });

        // Close drawers
        const drawers = document.querySelectorAll('.drawer.open');
        drawers.forEach(drawer => {
            drawer.classList.remove('open');
        });

        // Clear search input if focused
        const searchInput = document.getElementById('modelSearch');
        if (searchInput && document.activeElement === searchInput) {
            searchInput.value = '';
            searchInput.blur();
        }
    }

    showHelp() {
        const helpText = `
Keyboard Shortcuts:
─────────────────
Ctrl/Cmd + K : Focus search
Escape       : Close modals, exit inputs
?            : Show this help

Tab          : Navigate between elements
Shift + Tab  : Navigate backwards
Enter/Space  : Activate buttons/links
        `;

        // Remove existing help modal if present
        const existingHelp = document.getElementById('keyboard-help-modal');
        if (existingHelp) {
            existingHelp.remove();
            return;
        }

        const modal = document.createElement('div');
        modal.id = 'keyboard-help-modal';
        modal.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10000;
        `;

        const content = document.createElement('div');
        content.style.cssText = `
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 2rem;
            max-width: 400px;
            color: var(--text-primary);
            font-family: monospace;
            white-space: pre;
            line-height: 1.8;
        `;

        content.textContent = helpText;
        modal.appendChild(content);
        document.body.appendChild(modal);

        // Close on click or Escape
        modal.addEventListener('click', () => modal.remove());
        modal.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') modal.remove();
        });
    }

    setupAccessibility() {
        // Add ARIA labels to icon-only buttons
        const iconButtons = document.querySelectorAll('button:not([aria-label])');
        iconButtons.forEach(btn => {
            const text = btn.textContent.trim();
            if (text && !text.includes(' ')) {
                // Single character button (likely an icon)
                if (text === '☀️' || text === '🌙') {
                    btn.setAttribute('aria-label', 'Toggle theme');
                } else if (text === '⚒️') {
                    btn.setAttribute('aria-label', 'GGUF Forge Logo');
                }
            }
        });

        // Add skip to main content link
        if (!document.querySelector('.skip-to-content')) {
            const skipLink = document.createElement('a');
            skipLink.href = '#main-content';
            skipLink.className = 'skip-to-content';
            skipLink.textContent = 'Skip to main content';
            document.body.insertBefore(skipLink, document.body.firstChild);
        }

        // Add id to main content if not present
        const mainContent = document.querySelector('main');
        if (mainContent && !mainContent.id) {
            mainContent.id = 'main-content';
        }

        // Ensure all interactive elements have appropriate roles
        const interactiveElements = document.querySelectorAll('button, a, input, textarea, select');
        interactiveElements.forEach(el => {
            if (!el.getAttribute('tabindex')) {
                el.setAttribute('tabindex', '0');
            }
        });
    }
}

// Initialize keyboard manager
const keyboardManager = new KeyboardManager();
