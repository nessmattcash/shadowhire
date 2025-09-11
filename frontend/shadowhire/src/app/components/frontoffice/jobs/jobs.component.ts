import { Component, OnInit, OnDestroy, ViewChild, ElementRef, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router, RouterLink } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { HeaderComponent } from '../header/header.component';
import { FooterComponent } from '../footer/footer.component';
import { JobsService } from '../../../services/jobs.service';
import Swiper from 'swiper';
import { Navigation, Autoplay, EffectFade } from 'swiper/modules';
import AOS from 'aos';

interface Job {
  id: number;
  title: string;
  description: string;
  company: string;
  location: string;
  created_by: string;
  created_at: string;
  skills_required: string;
  featured?: boolean;
}

@Component({
  selector: 'app-jobs',
  standalone: true,
  imports: [CommonModule, RouterLink, FormsModule, HeaderComponent, FooterComponent],
  templateUrl: './jobs.component.html',
  styleUrls: ['./jobs.component.scss']
})
export class JobsComponent implements OnInit, OnDestroy, AfterViewInit {
  @ViewChild('companySwiper') companySwiperRef!: ElementRef;
  private companySwiper!: Swiper;
  
  jobs: Job[] = [];
  filteredJobs: Job[] = [];
  isLoading = true;
  error: string | null = null;
  searchTerm = '';
  skillsFilter = '';
  locationFilter = '';
  companyFilter = '';
  currentPage = 1;
  savedJobs: number[] = [];

  constructor(private jobsService: JobsService, private router: Router) {}

  ngOnInit(): void {
    AOS.init({ duration: 800, easing: 'ease-out' });
    this.loadSavedJobs();
    this.fetchJobs();
  }

  ngAfterViewInit(): void {
    this.initSwiper();
  }

  ngOnDestroy(): void {
    if (this.companySwiper) {
      this.companySwiper.destroy(true, true);
    }
  }

  private initSwiper(): void {
    this.companySwiper = new Swiper(this.companySwiperRef.nativeElement, {
      modules: [Navigation, Autoplay, EffectFade],
      loop: true,
      speed: 1000,
      autoplay: {
        delay: 10000,
        disableOnInteraction: false,
      },
      effect: 'fade',
      fadeEffect: {
        crossFade: true,
      },
      slidesPerView: 1,
      centeredSlides: true,
      grabCursor: true,
      navigation: {
        nextEl: '.swiper-button-next',
        prevEl: '.swiper-button-prev',
      },
      breakpoints: {
        768: {
          slidesPerView: 3,
          spaceBetween: 20,
        },
        1024: {
          slidesPerView: 4,
          spaceBetween: 30,
        },
      },
    });
  }

  fetchJobs(): void {
    this.isLoading = true;
    this.error = null;
    
    this.jobsService.getJobs().subscribe({
      next: (data: Job[]) => {
        this.jobs = data.map((job, index) => ({
          ...job,
          featured: index % 5 === 0
        }));
        this.applyFilters();
        this.isLoading = false;
      },
      error: (err: any) => {
        this.error = 'Failed to load jobs. Please try again later.';
        this.isLoading = false;
        console.error('Error fetching jobs:', err);
      }
    });
  }

  onSearchChange(event: any): void {
    this.searchTerm = event.target.value;
    this.applyFilters();
  }

  triggerSearch(): void {
    this.applyFilters();
  }

  applyFilters(): void {
    let results = [...this.jobs];
    
    if (this.searchTerm) {
      const term = this.searchTerm.toLowerCase();
      results = results.filter(job => 
        job.title.toLowerCase().includes(term) ||
        job.company.toLowerCase().includes(term) ||
        job.skills_required.toLowerCase().includes(term) ||
        job.description.toLowerCase().includes(term)
      );
    }
    
    if (this.skillsFilter) {
      const skills = this.skillsFilter.toLowerCase().split(',').map(s => s.trim());
      results = results.filter(job => 
        skills.some(skill => job.skills_required.toLowerCase().includes(skill))
      );
    }
    
    if (this.locationFilter) {
      results = results.filter(job => 
        job.location.toLowerCase().includes(this.locationFilter.toLowerCase())
      );
    }
    
    if (this.companyFilter) {
      results = results.filter(job => 
        job.company.toLowerCase().includes(this.companyFilter.toLowerCase())
      );
    }
    
    this.filteredJobs = results;
    this.currentPage = 1; // Reset to first page on filter
  }

  clearFilters(): void {
    this.searchTerm = '';
    this.skillsFilter = '';
    this.locationFilter = '';
    this.companyFilter = '';
    this.applyFilters();
  }

  onSortChange(event: any): void {
    const sortBy = event.target.value;
    
    switch (sortBy) {
      case 'newest':
        this.filteredJobs.sort((a, b) => 
          new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
        );
        break;
      case 'oldest':
        this.filteredJobs.sort((a, b) => 
          new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
        );
        break;
      case 'company':
        this.filteredJobs.sort((a, b) => 
          a.company.localeCompare(b.company)
        );
        break;
    }
    this.currentPage = 1; // Reset to first page on sort
  }

  getSkillsArray(skillsString: string): string[] {
    return skillsString.split(',').map(skill => skill.trim()).slice(0, 4);
  }

  formatDate(dateString: string): string {
    const date = new Date(dateString);
    const now = new Date();
    const diffTime = Math.abs(now.getTime() - date.getTime());
    const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));
    
    if (diffDays === 0) return 'Today';
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays} days ago`;
    if (diffDays < 30) return `${Math.floor(diffDays / 7)} weeks ago`;
    
    return date.toLocaleDateString();
  }

  getCompanyLogo(companyName: string): string {
    const logoMap: { [key: string]: string } = {
      'EY': '/assets/EY.jpg',
      'Capgemini': '/assets/cap.jpeg',
      'Microsoft': '/assets/micro.jpg',
      'ACTIA': '/assets/ACTIA.jpg',
      'Sopra Steria': '/assets/sopra.jpg',
      'OpenAI': '/assets/OpenAI.jpg'
    };
    
    return logoMap[companyName] || '/assets/default-company.png';
  }

  loadSavedJobs(): void {
    const saved = localStorage.getItem('savedJobs');
    if (saved) {
      this.savedJobs = JSON.parse(saved);
    }
  }

  toggleSaveJob(job: Job): void {
    const index = this.savedJobs.indexOf(job.id);
    
    if (index === -1) {
      this.savedJobs.push(job.id);
    } else {
      this.savedJobs.splice(index, 1);
    }
    
    localStorage.setItem('savedJobs', JSON.stringify(this.savedJobs));
  }

  isJobSaved(job: Job): boolean {
    return this.savedJobs.includes(job.id);
  }

  getPageNumbers(): number[] {
    return Array.from({ length: Math.ceil(this.filteredJobs.length / 6) }, (_, i) => i + 1);
  }

  goToPage(page: number): void {
    this.currentPage = page;
  }

  get paginatedJobs(): Job[] {
    const start = (this.currentPage - 1) * 6;
    const end = start + 6;
    return this.filteredJobs.slice(start, end);
  }

  getTwoLineDescription(description: string): string {
    const lines = description.split('\n').slice(0, 2).join(' ');
    return lines.length > 100 ? lines.substring(0, 100) + '...' : lines + '...';
  }

  navigateToJob(jobId: number): void {
    this.router.navigate(['/job', jobId]);
  }
}