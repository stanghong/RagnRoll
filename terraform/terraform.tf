provider "snowflake" {
  role = "SYSADMIN"
}

terraform {
  required_providers {
    snowflake = {
      source  = "Snowflake-Labs/snowflake"
      version = "~> 1.0.1"
    }
  }
}
