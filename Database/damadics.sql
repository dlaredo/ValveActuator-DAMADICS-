CREATE SCHEMA `damadics` DEFAULT CHARACTER SET utf8;

CREATE TABLE `damadics`.`valvereadings` (
  `id` BIGINT NOT NULL AUTO_INCREMENT,
  `timestamp` DATETIME NOT NULL,
  `externalControllerOutput` real NULL,
  `undisturbedMediumFlow` real NULL,
  `pressureValveInlet` real NULL,
  `pressureValveOutlet` real NULL,
  `mediumTemperature` real NULL,
  `rodDisplacement` real NULL,
  `disturbedMediumFlow` real NULL,
  `selectedFault` TINYINT NULL,
  `faultType` TINYINT NULL,
  `faultIntensity` real NULL,
  PRIMARY KEY (`id`),
  UNIQUE INDEX `timestamp_UNIQUE` (`timestamp` ASC));
