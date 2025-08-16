import { NgModule } from '@angular/core';
import { bootstrapApplication, BrowserModule } from '@angular/platform-browser';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { HttpClientModule } from '@angular/common/http';
import { APP_BASE_HREF, PlatformLocation } from '@angular/common';
import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';

// Standalone components
import { HomeComponent } from './components/frontoffice/home/home.component';
import { DashboardComponent } from './components/frontoffice/dashboard/dashboard.component';
import { FeedbackComponent } from './components/frontoffice/feedback/feedback.component';
import { JobsComponent } from './components/frontoffice/jobs/jobs.component';
import { ResumeUploadComponent } from './components/frontoffice/resume-upload/resume-upload.component';
import { JobDetailsComponent } from './components/frontoffice/job-details/job-details.component';
import { ApplyComponent } from './components/frontoffice/apply/apply.component';
import { LoginComponent } from './components/backoffice/login/login.component';
import { RegisterComponent } from './components/backoffice/register/register.component';
import { CandidateManagementComponent } from './components/backoffice/candidate-management/candidate-management.component';
import { MlFeedbackPanelComponent } from './components/backoffice/ml-feedback-panel/ml-feedback-panel.component';
import { OfferManagementComponent } from './components/backoffice/offer-management/offer-management.component';
import { ChatbotComponent } from './components/frontoffice/chatbot/chatbot.component';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';

@NgModule({
  declarations: [],
  imports: [
    BrowserModule,
    FormsModule,
    ReactiveFormsModule,
    HttpClientModule,
    AppRoutingModule,
    AppComponent,
    HomeComponent,
    DashboardComponent,
    FeedbackComponent,
    JobsComponent,
    ResumeUploadComponent,
    JobDetailsComponent,
    ApplyComponent,
    LoginComponent,
    RegisterComponent,
    CandidateManagementComponent,
    MlFeedbackPanelComponent,
    ChatbotComponent,
    OfferManagementComponent,
    BrowserAnimationsModule, // Enable animations
  ],
   providers: [
    {
      provide: APP_BASE_HREF,
      useFactory: (s: PlatformLocation) => {
        // This ensures no double slashes
        const base = s.getBaseHrefFromDOM();
        return base.endsWith('/') ? base.slice(0, -1) : base || '/';
      },
      deps: [PlatformLocation]
    }
  ],
  bootstrap: [AppComponent]
})
export class AppModule {}