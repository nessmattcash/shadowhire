import { Component, AfterViewInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-header',
  templateUrl: './header.component.html',
  styleUrls: ['./header.component.scss'],
  standalone: true,
  imports: [CommonModule],
})
export class HeaderComponent implements AfterViewInit, OnDestroy {
  ngAfterViewInit() {
    if (typeof window !== 'undefined') {
      // Mobile nav toggle
      const mobileNavToggle = document.querySelector('.mobile-nav-toggle') as HTMLElement;
      const navMenu = document.querySelector('#navmenu') as HTMLElement;
      if (mobileNavToggle && navMenu) {
        mobileNavToggle.addEventListener('click', () => {
          navMenu.classList.toggle('navmenu-active');
          mobileNavToggle.classList.toggle('bi-list');
          mobileNavToggle.classList.toggle('bi-x');
        });
      }

      // Navigation active state on scroll
      const navLinks = document.querySelectorAll('#navmenu a');
      const sections = document.querySelectorAll('section');
      const updateNav = () => {
        let current = '';
        sections.forEach((section) => {
          const sectionTop = section.offsetTop;
          const sectionHeight = section.offsetHeight;
          if (window.scrollY >= sectionTop - 60 && window.scrollY < sectionTop + sectionHeight) {
            current = section.getAttribute('id') || '';
          }
        });
        navLinks.forEach((link) => {
          link.classList.remove('active');
          if (link.getAttribute('href') === `#${current}`) {
            link.classList.add('active');
          }
        });
      };
      window.addEventListener('scroll', updateNav);
    }
  }

  ngOnDestroy() {
    // Clean up event listeners
    const mobileNavToggle = document.querySelector('.mobile-nav-toggle') as HTMLElement;
    if (mobileNavToggle) {
      mobileNavToggle.replaceWith(mobileNavToggle.cloneNode(true));
    }
    window.removeEventListener('scroll', () => {});
  }
}