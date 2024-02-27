terraform {
  required_providers {
    google = {
      source = "hashicorp/google"
      version = "4.51.0"
    }
  }
}

provider "google" {
  credentials = file("/Users/alexander.girardet/Code/Personal/projects/rightmove_project/credentials/airflow-service-account.json")

  project = "personal-projects-411616"
  region  = "europe-west2"
  zone    = "europe-west2-a"
}

#resource "google_compute_network" "vpc_network" {
#  name = "terraform-network"
#}
