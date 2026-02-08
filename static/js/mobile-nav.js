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
        this.createHamburgerButton();
        this.setupEventListeners();
        this.handleResize();
    }

    createHamburgerButton() {
        const navContent = document.querySelector('.nav-content');
        if (!navContent) return;

        // Check if hamburger already exists
        if (document.querySelector('.hamburger-btn')) return;

        const hamburger = document.createElement('button');
        hamburger.className = 'hamburger-btn';
        hamburger.setAttribute('aria-label', 'Toggle navigation');
        hamburger.setAttribute('aria-expanded', 'false');
        hamburger.innerHTML = `
            <span></span>
            <span></span>
            <span></span>
        `;

        hamburger.addEventListener('click', () => this.toggleNav());
        navContent.appendChild(hamburger);
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
        document.body.style.overflow = 'hidden'; // Prevent scrolling
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
        // Close mobile nav on resize to desktop
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
