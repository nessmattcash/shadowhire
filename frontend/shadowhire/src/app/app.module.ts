import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { provideHttpClient, withInterceptorsFromDi } from '@angular/common/http';
import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
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

@NgModule({
  declarations: [], // Remove AppComponent from declarations
  imports: [
    BrowserModule,
    FormsModule,
    ReactiveFormsModule,
    AppRoutingModule,
    AppComponent, // Add AppComponent to imports
    // Front office components
    HomeComponent,
    DashboardComponent,
    FeedbackComponent,
    JobsComponent,
    ResumeUploadComponent,
    JobDetailsComponent,
    ApplyComponent,
    // Back office components
    LoginComponent,
    RegisterComponent,
    CandidateManagementComponent,
    MlFeedbackPanelComponent,
    OfferManagementComponent,
  ],
  providers: [
    provideHttpClient(withInterceptorsFromDi()),
  ],
  bootstrap: [AppComponent],
})
export class AppModule {}