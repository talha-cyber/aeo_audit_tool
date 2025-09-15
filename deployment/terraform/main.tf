terraform {
  required_version = ">= 1.5.0"
}

provider "aws" {
  region = var.region
}

# Placeholder: define VPC, ECS/EKS, or EC2 infra here

output "note" {
  value = "Terraform placeholder for infra as code."
}
