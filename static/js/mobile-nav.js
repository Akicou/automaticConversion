/**
 * Mobile Navigation
 */

class MobileNavManager {
    constructor() {
        this.mobileBreakpoint = 768;
        this.navOpen = false;
        this.backdrop = null;
        this.init();
    }

    init() {
        this.createBackdrop();
        this.setupHamburgerButton();
        this.setupEventListeners();
    }

    createBackdrop() {
        // Create backdrop element for closing drawer
        this.backdrop = document.createElement('div');
        this.backdrop.className = 'nav-backdrop';
        this.backdrop.addEventListener('click', () => this.closeNav());
        document.body.appendChild(this.backdrop);
    }

    setupHamburgerButton() {
        const hamburger = document.querySelector('.hamburger-btn');
        if (!hamburger) return;
        hamburger.addEventListener('click', () => this.toggleNav());
    }

    setupEventListeners() {
        // Close on resize to desktop
        window.addEventListener('resize', () => {
            if (window.innerWidth >= this.mobileBreakpoint && this.navOpen) {
                this.closeNav();
            }
        });

        // Close on nav link click
        document.querySelectorAll('.nav-item, .nav-btn').forEach(link => {
            link.addEventListener('click', () => {
                if (window.innerWidth < this.mobileBreakpoint && this.navOpen) {
                    this.closeNav();
                }
            });
        });
    }

    toggleNav() {
        this.navOpen ? this.closeNav() : this.openNav();
    }

    openNav() {
        const navLinks = document.querySelector('.nav-links');
        const hamburger = document.querySelector('.hamburger-btn');

        navLinks?.classList.add('nav-links-open');
        hamburger?.classList.add('hamburger-open');
        this.backdrop?.classList.add('active');
        document.body.style.overflow = 'hidden';
        this.navOpen = true;
    }

    closeNav() {
        const navLinks = document.querySelector('.nav-links');
        const hamburger = document.querySelector('.hamburger-btn');

        navLinks?.classList.remove('nav-links-open');
        hamburger?.classList.remove('hamburger-open');
        this.backdrop?.classList.remove('active');
        document.body.style.overflow = '';
        this.navOpen = false;
    }
}

// Initialize
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => new MobileNavManager());
} else {
    new MobileNavManager();
}
