/*
Navicat MySQL Data Transfer

Source Server         : MySql
Source Server Version : 50723
Source Host           : 127.0.0.1:3306
Source Database       : siw

Target Server Type    : MYSQL
Target Server Version : 50723
File Encoding         : 65001

Date: 2018-11-15 19:35:49
*/

SET FOREIGN_KEY_CHECKS=0;

-- ----------------------------
-- Table structure for tb_file
-- ----------------------------
DROP TABLE IF EXISTS `tb_file`;
CREATE TABLE `tb_file` (
  `file_id` int(20) NOT NULL AUTO_INCREMENT,
  `file_name` varchar(255) NOT NULL,
  `file_type` varchar(10) DEFAULT NULL,
  `file_date` datetime DEFAULT NULL,
  `file_size` varchar(50) DEFAULT NULL,
  `file_path` varchar(255) NOT NULL,
  PRIMARY KEY (`file_id`)
) ENGINE=InnoDB AUTO_INCREMENT=5 DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of tb_file
-- ----------------------------
INSERT INTO `tb_file` VALUES ('1', '权利的游戏(第二季)', 'txt', '2018-11-15 16:32:32', '4.8K', '/Data/');
INSERT INTO `tb_file` VALUES ('2', '权利的游戏(第三季)', 'txt', '2018-11-15 16:32:32', '4.316K', '/Data/');
INSERT INTO `tb_file` VALUES ('3', '权利的游戏(第一季)', 'txt', '2018-11-15 16:32:32', '7.32K', '/Data/');
INSERT INTO `tb_file` VALUES ('4', '自来水表移交', 'jpg', '2018-11-15 18:41:22', '1805.326K', '/Data/');

-- ----------------------------
-- Table structure for tb_user
-- ----------------------------
DROP TABLE IF EXISTS `tb_user`;
CREATE TABLE `tb_user` (
  `user_id` varchar(50) NOT NULL,
  `user_name` varchar(50) NOT NULL,
  `user_password` varchar(50) NOT NULL,
  `user_description` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`user_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of tb_user
-- ----------------------------
INSERT INTO `tb_user` VALUES ('admin', '管理员', 'admin', '管理员');
INSERT INTO `tb_user` VALUES ('test', '测试员', 'test', '测试员');
