/**
 * Toast Notification System
 * Provides non-intrusive user feedback with success, error, warning, and info variants
 */

class ToastManager {
    constructor() {
        this.container = null;
        this.toasts = [];
        this.maxToasts = 4;
        this.defaultDuration = 4000;
        this.init();
    }

    init() {
        // Create toast container if it doesn't exist
        this.container = document.getElementById('toast-container');
        if (!this.container) {
            this.container = document.createElement('div');
            this.container.id = 'toast-container';
            document.body.appendChild(this.container);
        }
    }

    /**
     * Show a toast notification
     * @param {string} type - Type of toast: 'success', 'error', 'warning', 'info'
     * @param {string} message - Main message content
     * @param {Object} options - Additional options
     * @param {string} options.title - Optional title
     * @param {number} options.duration - Duration in ms (0 for no auto-dismiss)
     * @param {boolean} options.closeable - Show close button (default: true)
     */
    show(type, message, options = {}) {
        const {
            title = null,
            duration = this.defaultDuration,
            closeable = true
        } = options;

        // Remove oldest toast if we've hit the limit
        if (this.toasts.length >= this.maxToasts) {
            this.remove(this.toasts[0].id);
        }

        const toastId = 'toast-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
        const toast = this.createToastElement(toastId, type, message, title, closeable, duration);

        this.container.appendChild(toast);
        this.toasts.push({ id: toastId, element: toast });

        // Auto-dismiss after duration (if duration > 0)
        if (duration > 0) {
            setTimeout(() => this.remove(toastId), duration);
        }

        return toastId;
    }

    createToastElement(id, type, message, title, closeable, duration) {
        const toast = document.createElement('div');
        toast.id = id;
        toast.className = `toast toast-${type}`;
        toast.setAttribute('role', 'alert');
        toast.setAttribute('aria-live', 'polite');

        const icon = this.getIcon(type);

        let html = `
            <div class="toast-icon">${icon}</div>
            <div class="toast-content">
        `;

        if (title) {
            html += `<div class="toast-title">${this.escapeHtml(title)}</div>`;
        }

        html += `
                <div class="toast-message">${this.escapeHtml(message)}</div>
            </div>
        `;

        if (closeable) {
            html += `<button class="toast-close" onclick="toastManager.remove('${id}')" aria-label="Close notification">&times;</button>`;
        }

        if (duration > 0) {
            html += `<div class="toast-progress" style="animation-duration: ${duration}ms;"></div>`;
        }

        toast.innerHTML = html;
        return toast;
    }

    getIcon(type) {
        const icons = {
            success: '✓',
            error: '✕',
            warning: '⚠',
            info: 'ℹ'
        };
        return icons[type] || icons.info;
    }

    remove(toastId) {
        const toastIndex = this.toasts.findIndex(t => t.id === toastId);
        if (toastIndex === -1) return;

        const toast = this.toasts[toastIndex];
        toast.element.classList.add('toast-removing');

        setTimeout(() => {
            if (toast.element.parentNode) {
                toast.element.parentNode.removeChild(toast.element);
            }
            this.toasts.splice(toastIndex, 1);
        }, 300); // Match animation duration
    }

    clear() {
        this.toasts.forEach(toast => {
            toast.element.classList.add('toast-removing');
        });

        setTimeout(() => {
            this.container.innerHTML = '';
            this.toasts = [];
        }, 300);
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Create global toast manager instance
const toastManager = new ToastManager();

// Convenience functions
function showToast(type, message, options = {}) {
    return toastManager.show(type, message, options);
}

function showSuccess(message, options = {}) {
    return showToast('success', message, options);
}

function showError(message, options = {}) {
    return showToast('error', message, options);
}

function showWarning(message, options = {}) {
    return showToast('warning', message, options);
}

function showInfo(message, options = {}) {
    return showToast('info', message, options);
}
