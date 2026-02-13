/**
 * Mobile Navigation
 * Handles hamburger menu and mobile-specific navigation features
 */

class MobileNavManager {
    constructor() {
        this.mobileBreakpoint = 768;
        this.navOpen = false;
        this.init();
    }

    init() {
        this.setupHamburgerButton();
        this.setupEventListeners();
        this.handleResize();
    }

    setupHamburgerButton() {
        const hamburger = document.querySelector('.hamburger-btn');
        if (!hamburger) return;
        hamburger.addEventListener('click', () => this.toggleNav());
    }

    setupEventListeners() {
        // Close mobile nav when clicking outside
        document.addEventListener('click', (e) => {
            if (this.navOpen && !e.target.closest('.nav-links') && !e.target.closest('.hamburger-btn')) {
                this.closeNav();
            }
        });

        // Close mobile nav on window resize
        window.addEventListener('resize', () => this.handleResize());

        // Handle mobile nav links
        const navLinks = document.querySelectorAll('.nav-item, .nav-btn');
        navLinks.forEach(link => {
            link.addEventListener('click', () => {
                if (window.innerWidth < this.mobileBreakpoint) {
                    this.closeNav();
                }
            });
        });
    }

    toggleNav() {
        if (this.navOpen) {
            this.closeNav();
        } else {
            this.openNav();
        }
    }

    openNav() {
        const navLinks = document.querySelector('.nav-links');
        const hamburger = document.querySelector('.hamburger-btn');

        if (navLinks) {
            navLinks.classList.add('nav-links-open');
        }
        if (hamburger) {
            hamburger.classList.add('hamburger-open');
            hamburger.setAttribute('aria-expanded', 'true');
        }

        this.navOpen = true;
        document.body.style.overflow = 'hidden';
    }

    closeNav() {
        const navLinks = document.querySelector('.nav-links');
        const hamburger = document.querySelector('.hamburger-btn');

        if (navLinks) {
            navLinks.classList.remove('nav-links-open');
        }
        if (hamburger) {
            hamburger.classList.remove('hamburger-open');
            hamburger.setAttribute('aria-expanded', 'false');
        }

        this.navOpen = false;
        document.body.style.overflow = '';
    }

    handleResize() {
        if (window.innerWidth >= this.mobileBreakpoint && this.navOpen) {
            this.closeNav();
        }
    }
}

// Initialize mobile navigation when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        new MobileNavManager();
    });
} else {
    new MobileNavManager();
}
