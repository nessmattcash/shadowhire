import { Component, AfterViewInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HeaderComponent } from '../header/header.component';
import { FooterComponent } from '../footer/footer.component';
import { ChatbotComponent } from '../chatbot/chatbot.component';
import { ChatbotService } from '../../../services/chatbot.service';

declare const AOS: any;
declare const Swiper: any;
declare const GLightbox: any;
declare const Isotope: any;

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.scss'],
  standalone: true,
  imports: [CommonModule, HeaderComponent, FooterComponent, ChatbotComponent]
})
export class HomeComponent implements AfterViewInit, OnDestroy {
  constructor(private chatbotService: ChatbotService) {}
  infoEmail = 'info@example.com';
  contactEmail = 'contact@example.com';

  toggleChatbot(): void {
    this.chatbotService.toggleChat();
  }

  ngAfterViewInit() {
    if (typeof window !== 'undefined') {
      // Remove preloader
      const preloader = document.querySelector('#preloader') as HTMLElement;
      if (preloader) preloader.remove();

      // Initialize AOS
      if (typeof AOS !== 'undefined') {
        AOS.init({
          duration: 800,
          easing: 'ease-in-out',
          once: true,
          mirror: false
        });
      }

      // Initialize Swiper
      if (typeof Swiper !== 'undefined') {
        document.querySelectorAll('.init-swiper').forEach((swiperElement) => {
          const configElement = swiperElement.querySelector('.swiper-config') as HTMLElement;
          const config = configElement ? JSON.parse(configElement.innerHTML) : {};
          new Swiper(swiperElement, config);
        });
      }

      // Initialize GLightbox
      if (typeof GLightbox !== 'undefined') {
        GLightbox({
          selector: '.glightbox'
        });
      }

      // Initialize Isotope
      if (typeof Isotope !== 'undefined') {
        const isotopeContainer = document.querySelector('.isotope-container') as HTMLElement;
        if (isotopeContainer) {
          const isotope = new Isotope(isotopeContainer, {
            itemSelector: '.isotope-item',
            layoutMode: 'masonry'
          });
          const filters = document.querySelectorAll('.isotope-filters li');
          filters.forEach((filter) => {
            filter.addEventListener('click', () => {
              filters.forEach((el) => el.classList.remove('filter-active'));
              filter.classList.add('filter-active');
              isotope.arrange({
                filter: filter.getAttribute('data-filter')
              });
            });
          });
        }
      }

      // FAQ Toggle
      const faqItems = document.querySelectorAll('.faq-item');
      faqItems.forEach((item) => {
        const toggle = item.querySelector('.faq-toggle') as HTMLElement;
        if (toggle) {
          toggle.addEventListener('click', () => {
            item.classList.toggle('faq-active');
          });
        }
      });
    }
  }

  ngOnDestroy() {
    // Clean up event listeners
    const faqItems = document.querySelectorAll('.faq-item');
    faqItems.forEach((item) => {
      const toggle = item.querySelector('.faq-toggle') as HTMLElement;
      if (toggle) toggle.replaceWith(toggle.cloneNode(true));
    });
    if (typeof AOS !== 'undefined') AOS.refresh();
  }
}