import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';

// Front office components
import { HomeComponent } from './components/frontoffice/home/home.component';
import { DashboardComponent } from './components/frontoffice/dashboard/dashboard.component';
import { FeedbackComponent } from './components/frontoffice/feedback/feedback.component';
import { JobsComponent } from './components/frontoffice/jobs/jobs.component';
import { ResumeUploadComponent } from './components/frontoffice/resume-upload/resume-upload.component';
import { JobDetailsComponent } from './components/frontoffice/job-details/job-details.component';
import { ApplyComponent } from './components/frontoffice/apply/apply.component';

// Back office components
import { LoginComponent } from './components/backoffice/login/login.component';
import { RegisterComponent } from './components/backoffice/register/register.component';
import { CandidateManagementComponent } from './components/backoffice/candidate-management/candidate-management.component';
import { MlFeedbackPanelComponent } from './components/backoffice/ml-feedback-panel/ml-feedback-panel.component';
import { OfferManagementComponent } from './components/backoffice/offer-management/offer-management.component';

const routes: Routes = [
  // Front office routes
  { path: '', component: HomeComponent },
  { path: 'dashboard', component: DashboardComponent },
  { path: 'feedback', component: FeedbackComponent },
  { path: 'jobs', component: JobsComponent },
  { path: 'resume-upload', component: ResumeUploadComponent },
  { path: 'job-details/:id', component: JobDetailsComponent }, // assuming id param
  { path: 'apply/:id', component: ApplyComponent },           // assuming id param

  // Back office routes
  { path: 'login', component: LoginComponent },
  { path: 'register', component: RegisterComponent },
  { path: 'candidate-management', component: CandidateManagementComponent },
  { path: 'ml-feedback-panel', component: MlFeedbackPanelComponent },
  { path: 'offer-management', component: OfferManagementComponent },

  // Wildcard route, redirect to home
  { path: '**', redirectTo: '', pathMatch: 'full' }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule],
})
export class AppRoutingModule {}
