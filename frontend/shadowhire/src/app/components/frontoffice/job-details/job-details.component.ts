import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule, NgForm } from '@angular/forms';
import { ActivatedRoute, Router } from '@angular/router';
import { HeaderComponent } from '../header/header.component';
import { FooterComponent } from '../footer/footer.component';
import { JobsService } from '../../../services/jobs.service';
import AOS from 'aos';

interface Application {
  name: string;
  email: string;
  resume?: File;
  coverLetter: string;
}
interface Job {
  id: number;
  title: string;
  description: string;
  job_overview: string;
  responsibilities: string;
  company: string;
  location: string;
  created_by: string;
  created_at: string;
  skills_required: string;
  benefits: string;
  job_type: string;
  qualification_level: string;
  featured?: boolean;
  isFeaturedShown?: boolean;
}


@Component({
  selector: 'app-job-details',
  standalone: true,
  imports: [CommonModule, FormsModule, HeaderComponent, FooterComponent],
  templateUrl: './job-details.component.html',
  styleUrls: ['./job-details.component.scss']
})
export class JobDetailsComponent implements OnInit, OnDestroy {
  job: Job | null = null;
  relatedJobs: Job[] = [];
  showApplyModal = false;
  application: Application = { name: '', email: '', coverLetter: '' };
  savedJobs: number[] = [];
  userSkills: string[] = ['JavaScript', 'React', 'Angular']; // Mock user skills for matching

  constructor(
    private route: ActivatedRoute,
    private router: Router,
    private jobsService: JobsService
  ) {}

  ngOnInit(): void {
    AOS.init({ duration: 800, easing: 'ease-out' });
    this.loadSavedJobs();
    const jobId = Number(this.route.snapshot.paramMap.get('id'));
    this.fetchJobDetails(jobId);
  }

  ngOnDestroy(): void {}

  fetchJobDetails(id: number): void {
    this.jobsService.getJobById(id).subscribe({
      next: (job: Job) => {
        this.job = { ...job, featured: true, isFeaturedShown: this.isJobSaved(job) };
        this.fetchRelatedJobs();
      },
      error: (err) => {
        console.error('Error fetching job details:', err);
        this.router.navigate(['/jobs']);
      }
    });
  }

  fetchRelatedJobs(): void {
    if (!this.job) return;
    // Ideally, call backend API /jobs/related/<company>/
    // Mock implementation for now
    this.jobsService.getJobs().subscribe({
      next: (jobs: Job[]) => {
        this.relatedJobs = jobs
          .filter(job => job.company === this.job!.company && job.id !== this.job!.id)
          .slice(0, 2); // Limit to 2 related jobs
        if (this.relatedJobs.length === 0) {
          // Fallback mock data
          this.relatedJobs = [
            {
              id: this.job!.id + 1,
              title: `Senior ${this.job!.title.split(' ')[0]} Developer`,
              description: this.job!.description,
              job_overview: `Advanced role at ${this.job!.company} focusing on ${this.job!.skills_required}.`,
              responsibilities: this.job!.responsibilities,
              company: this.job!.company,
              location: this.job!.location === 'Remote' ? 'New York' : this.job!.location,
              created_by: this.job!.created_by,
              created_at: new Date().toISOString(),
              skills_required: `Advanced ${this.job!.skills_required}`,
              benefits: this.job!.benefits,
              job_type: this.job!.job_type,
              qualification_level: this.job!.qualification_level === 'senior' ? 'mid' : 'senior',
              featured: false,
              isFeaturedShown: false
            },
            {
              id: this.job!.id + 2,
              title: `Junior ${this.job!.title.split(' ')[0]} Specialist`,
              description: this.job!.description,
              job_overview: `Entry-level position at ${this.job!.company} with ${this.job!.skills_required}.`,
              responsibilities: this.job!.responsibilities,
              company: this.job!.company,
              location: 'Remote',
              created_by: this.job!.created_by,
              created_at: new Date().toISOString(),
              skills_required: `Basic ${this.job!.skills_required}`,
              benefits: this.job!.benefits,
              job_type: 'remote',
              qualification_level: 'junior',
              featured: false,
              isFeaturedShown: false
            }
          ];
        }
      },
      error: (err) => {
        console.error('Error fetching related jobs:', err);
        // Fallback to mock data if API fails
        this.relatedJobs = [
          {
            id: this.job!.id + 1,
            title: `Senior ${this.job!.title.split(' ')[0]} Developer`,
            description: this.job!.description,
            job_overview: `Advanced role at ${this.job!.company} focusing on ${this.job!.skills_required}.`,
            responsibilities: this.job!.responsibilities,
            company: this.job!.company,
            location: this.job!.location === 'Remote' ? 'New York' : this.job!.location,
            created_by: this.job!.created_by,
            created_at: new Date().toISOString(),
            skills_required: `Advanced ${this.job!.skills_required}`,
            benefits: this.job!.benefits,
            job_type: this.job!.job_type,
            qualification_level: this.job!.qualification_level === 'senior' ? 'mid' : 'senior',
            featured: false,
            isFeaturedShown: false
          },
          {
            id: this.job!.id + 2,
            title: `Junior ${this.job!.title.split(' ')[0]} Specialist`,
            description: this.job!.description,
            job_overview: `Entry-level position at ${this.job!.company} with ${this.job!.skills_required}.`,
            responsibilities: this.job!.responsibilities,
            company: this.job!.company,
            location: 'Remote',
            created_by: this.job!.created_by,
            created_at: new Date().toISOString(),
            skills_required: `Basic ${this.job!.skills_required}`,
            benefits: this.job!.benefits,
            job_type: 'remote',
            qualification_level: 'junior',
            featured: false,
            isFeaturedShown: false
          }
        ];
      }
    });
  }

  applyForJob(): void {
    this.showApplyModal = true;
  }

  closeApplyModal(event?: Event): void {
    if (event) event.preventDefault();
    this.showApplyModal = false;
  }

  stopPropagation(event: Event): void {
    event.stopPropagation();
  }

  onFileChange(event: any): void {
    this.application.resume = event.target.files[0];
  }

  submitApplication(form: NgForm): void {
    if (form.valid && this.application.resume) {
      // POST to backend /applications/ with job.id, user token, etc.
      console.log('Application submitted:', this.application, 'for job:', this.job?.id);
      alert('Application submitted successfully!');
      this.closeApplyModal();
      form.resetForm();
      this.application = { name: '', email: '', coverLetter: '' };
    } else {
      alert('Please fill all required fields and select a resume.');
    }
  }

  visitCompanyWebsite(): void {
    // Mock; replace with actual company URL from backend
    window.open('https://example.com/' + this.job?.company?.toLowerCase().replace(/\s+/g, '-'), '_blank');
  }

  navigateToRelatedJob(id: number): void {
    this.router.navigate(['/job-details', id]);
  }

  getSkillsArray(skillsString: string): string[] {
    return skillsString.split(',').map(skill => skill.trim());
  }

  isSkillMatched(skill: string): boolean {
    return this.userSkills.some(userSkill => userSkill.toLowerCase() === skill.toLowerCase());
  }

  getResponsibilities(responsibilities: string): string[] {
    if (!responsibilities) return [];
    return responsibilities.split('\n').map(s => s.trim()).filter(s => s.length > 0);
  }

  getBenefits(benefits: string): string[] {
    if (!benefits) return ['Competitive salary', 'Flexible working hours', 'Professional development opportunities'];
    return benefits.split('\n').map(s => s.trim()).filter(s => s.length > 0);
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

  formatJobType(type: string): string {
    return type.charAt(0).toUpperCase() + type.slice(1);
  }

  formatQualification(level: string): string {
    return level.charAt(0).toUpperCase() + level.slice(1) + ' Level';
  }

  getCompanyLogo(companyName: string): string {
    const companyLogos: { [key: string]: string } = {
      'EY': '/assets/EY.jpg',
      'Actia': '/assets/ACTIA.jpg',
      'Sagemcom': '/assets/sagemcom.jpg',
      'Defensy': '/assets/def.png',
      'Verse': '/assets/verse.jpg.png',
      'Sopra Steria': '/assets/sopra.jpg',
      'Orange': '/assets/orange.jpg',
      'Ooredoo': '/assets/ooredoo.jpg',
      'InstaDeep': '/assets/insta.png',
      'RFC': '/assets/rfc.png',
      'Vermeg': '/assets/vermeg.jpg',
      'Expensya': '/assets/exp.jpg',
      'Sofrecom': '/assets/sofrecom.jpg',
      'Talan': '/assets/talan.jpg',
      'NeoSoft': '/assets/neo.jpg',
      'Amaris': '/assets/amaris.jpg',
      'Business & Decision': '/assets/bd.png',
      'Telnet': '/assets/telnet.jpg',
      'Focus': '/assets/focus.jpg',
      'IBM': '/assets/ibm.jpg',
      'Oradist': '/assets/images.jpg',
      'Wattnow': '/assets/wat.png',
      'Dabchy': '/assets/dabchy.png',
      'Roamsmart': '/assets/roam.jpg',
      'Capgemini': '/assets/cap.jpeg'
    };
    return companyLogos[companyName] || '/assets/default-logo.jpg';
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
      job.isFeaturedShown = true;
    } else {
      this.savedJobs.splice(index, 1);
      job.isFeaturedShown = false;
    }
    localStorage.setItem('savedJobs', JSON.stringify(this.savedJobs));
  }

  isJobSaved(job: Job): boolean {
    return this.savedJobs.includes(job.id);
  }
}