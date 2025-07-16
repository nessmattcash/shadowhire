import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { HomeComponent } from './components/home/home.component';
import { LoginComponent } from './components/login/login.component';
import { RegisterComponent } from './components/register/register.component';
import { DashboardComponent } from './components/dashboard/dashboard.component';
import { ResumeUploadComponent } from './components/resume-upload/resume-upload.component';
import { JobsComponent } from './components/jobs/jobs.component';
import { AdminPanelComponent } from './components/admin-panel/admin-panel.component';
import { ChatbotComponent } from './components/chatbot/chatbot.component';
import { FeedbackComponent } from './components/feedback/feedback.component';

const routes: Routes = [
  { path: '', component: HomeComponent },
  { path: 'login', component: LoginComponent },
  { path: 'register', component: RegisterComponent },
  { path: 'dashboard', component: DashboardComponent },
  { path: 'resume-upload', component: ResumeUploadComponent },
  { path: 'jobs', component: JobsComponent },
  { path: 'admin-panel', component: AdminPanelComponent },
  { path: 'chatbot', component: ChatbotComponent },
  { path: 'feedback', component: FeedbackComponent },
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule],
})
export class AppRoutingModule {}
